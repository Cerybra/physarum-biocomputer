#![no_std]
#![no_main]

mod datasets;
mod drivers;
mod linalg;
mod random;

use core::f32::consts::PI;
use core::fmt::Write;

use libm::{exp, fabsf, floorf, sin, sqrtf};

use panic_halt as _;

use cortex_m_semihosting::hprintln;

use embedded_hal::blocking::spi::WriteIter;
use embedded_hal::serial::{Read as ReadSerial, Write as WriteSerial};

use stm32h7xx_hal::{nb, prelude::*, serial::Error, spi};

use stm32h7xx_hal::rng::RngCore;

use crate::drivers::adc::{ParallelBus, ParallelBusConfig, ADC};
use crate::drivers::dac::DAC;

use crate::linalg::{Array, Dot, Vector};

use crate::random::distribution::Gaussian;
use crate::random::generator::{Generator, XORShift32};

const SEED: u32 = 237643274;

// NOTE: There is a bug where this hangs exactly at 1204.
const STEPS: usize = 1200;
const SPLIT: usize = 1000;

const HORIZON: usize = 5;

const PHYSICAL_NODES: usize = 8;
const VIRTUAL_NODES: usize = 4;

const STATES: usize = 2 * VIRTUAL_NODES * PHYSICAL_NODES;
const DIMENSION: usize = STATES + 4;

#[link_section = ".axisram"]
// The last 32 are reservoir recordings, while the first four are the coordinates and target, respectively.
static mut RECORDINGS: [[f32; DIMENSION]; STEPS] = [[0.0; DIMENSION]; STEPS];

pub type VanDerPolDataset<const N: usize> = [(f32, f32, f32); N];
pub type RabinovichFabrikant<const N: usize> = [(f32, f32, f32); N];
pub type Rossler<const N: usize> = [(f32, f32, f32); N];

pub fn van_der_pol<const N: usize>(mu: f32, dt: f32) -> impl Iterator<Item = (f32, f32, f32)> {
    let (mut x, mut y): (f32, f32) = (1.0, 0.0);

    (0..N).into_iter().map(move |t| {
        let phase = (t as f32 / (N as f32 - 1.0)) * 15.0 * PI;
        let force = 0.5 * libm::sinf(phase);

        let point = (x, y, force);

        let dx_dt = y;
        let dy_dt = mu * (1.0 - (x * x)) * y - x + force;

        x += dx_dt * dt;
        y += dy_dt * dt;

        point
    })
}

pub fn rabinovich_fabrikant<const N: usize>(alpha: f32, gamma: f32, dt: f32) -> impl Iterator<Item = (f32, f32, f32)> {
    let (mut x, mut y, mut z) = (-1.0, 0.0, 0.5);

    (0..N).into_iter().map(move |t| {
        let point = (x, y, z);

        let dx_dt = y * (z - 1.0 + (x * x)) + gamma * x;
        let dy_dt = x * (3.0 * z + 1.0 - (x * x)) + gamma * y;
        let dz_dt = -2.0 * z * (alpha + x * y);

        x += dx_dt * dt;
        y += dy_dt * dt;
        z += dz_dt * dt;

        point
    })
}

pub fn rossler<const N: usize>(a: f32, b: f32, c: f32, dt: f32) -> impl Iterator<Item = (f32, f32, f32)> {
    // a, b, c = 0.2, 0.2, 5.7.
    let (mut x, mut y, mut z) = (0.1, 0.0, 0.0);

    (0..N).into_iter().map(move |t| {
        let point = (x, y, z);

        let dx_dt = -y - z;
        let dy_dt = x + (a * y);
        let dz_dt = b + (z * (x - c));

        x += dx_dt * dt;
        y += dy_dt * dt;
        z += dz_dt * dt;

        point
    })
}

pub fn train<const S: usize, const N: usize>(
    recordings: &[[f32; S]],
    readout: &mut Vector<N>,
    lr: f32,
    baseline: bool,
) {
    for t in 0..recordings.len() {
        let (points, target) = ((recordings[t][0], recordings[t][2]), recordings[t][3]);

        let state: Vector<N> = if baseline {
            let mut s = [0.0; N];
            if N >= 2 {
                s[0] = points.0;
                s[1] = points.1;
            }
            Vector::from(s)
        } else {
            let mut s = [0.0; N];
            s.copy_from_slice(&recordings[t][(S - N)..]);

            Vector::from(s)
        };

        let error = target - state.dot(&mut *readout);

        *readout = *readout + (state * (lr * error));
    }
}

pub fn test<const S: usize, const N: usize>(
    recordings: &[[f32; S]],
    readout: &Vector<N>,
    baseline: bool,
) -> f32 {
    let mut errors = 0.0;

    for t in 0..recordings.len() {
        let (points, target) = ((recordings[t][0], recordings[t][2]), recordings[t][3]);

        let state: Vector<N> = if baseline {
            let mut s = [0.0; N];
            if N >= 2 {
                s[0] = points.0;
                s[1] = points.1;
            }
            Vector::from(s)
        } else {
            let mut s = [0.0; N];
            s.copy_from_slice(&recordings[t][(S - N)..]);

            Vector::from(s)
        };

        errors += libm::powf(state.dot(readout) - target, 2.0);
    }

    errors / (recordings.len() as f32)
}

#[cortex_m_rt::entry]
fn main() -> ! {
    hprintln!("Initialized...");

    // Get access to the core peripherals.
    let mut core = cortex_m::Peripherals::take().unwrap();

    // Get access to the device specific peripherals from the peripheral access crate.
    let dp = stm32h7xx_hal::stm32::Peripherals::take().unwrap();

    // Take ownership over the RCC devices and convert them into the corresponding HAL structs.
    let rcc = dp.RCC.constrain();

    let pwr = dp.PWR.constrain();
    let pwrcfg = pwr.freeze();

    // Freeze the configuration of all the clocks in the system and
    // retrieve the Core Clock Distribution and Reset (CCDR) object.
    let rcc = rcc.sys_ck(400.MHz()).use_hse(8.MHz()).bypass_hse();
    let ccdr = rcc.freeze(pwrcfg, &dp.SYSCFG);

    let mut delay = core.SYST.delay(ccdr.clocks);

    // Acquire the GPIO peripherals.
    let gpio_a = dp.GPIOA.split(ccdr.peripheral.GPIOA);
    let gpio_b = dp.GPIOB.split(ccdr.peripheral.GPIOB);
    let gpio_d = dp.GPIOD.split(ccdr.peripheral.GPIOD);
    let gpio_e = dp.GPIOE.split(ccdr.peripheral.GPIOE);
    let gpio_f = dp.GPIOF.split(ccdr.peripheral.GPIOF);

    // Configuring the pins for the ADC parallel interface.
    let config = ParallelBusConfig {
        db0: gpio_e.pe3.into_pull_down_input(),
        db1: gpio_b.pb1.into_pull_down_input(),
        db2: gpio_b.pb2.into_pull_down_input(),
        db3: gpio_b.pb3.into_pull_down_input(),
        db4: gpio_b.pb4.into_pull_down_input(),
        db5: gpio_b.pb5.into_pull_down_input(),
        db6: gpio_b.pb6.into_pull_down_input(),
        db7: gpio_b.pb7.into_pull_down_input(),
        db8: gpio_b.pb8.into_pull_down_input(),
        db9: gpio_b.pb9.into_pull_down_input(),
        db10: gpio_b.pb10.into_pull_down_input(),
        db11: gpio_b.pb11.into_pull_down_input(),
        db12: gpio_b.pb12.into_pull_down_input(),
        db13: gpio_e.pe4.into_pull_down_input(),
        db14: gpio_e.pe2.into_pull_down_input(),
        db15: gpio_b.pb15.into_pull_down_input(),
    };

    let convst = gpio_d.pd3.into_push_pull_output();

    let eoc = gpio_d.pd4.into_pull_down_input();

    let rd = gpio_d.pd6.into_push_pull_output();
    let wr = gpio_d.pd7.into_push_pull_output();

    let sck = gpio_f.pf7.into_alternate();
    let miso = gpio_f.pf8.into_alternate();
    let mosi = gpio_f.pf9.into_alternate();

    // Initialize required CS pins.
    let dac_cs = gpio_a.pa0.into_push_pull_output();
    let adc_cs = gpio_d.pd5.into_push_pull_output();

    hprintln!("Initializing SPI ... ");

    // Initialize the SPI peripheral.
    let mut spi = dp.SPI5.spi(
        (sck, miso, mosi),
        spi::MODE_0,
        3.MHz(),
        ccdr.peripheral.SPI5,
        &ccdr.clocks,
    );

    // Initialize serial.
    let tx = gpio_d.pd8.into_alternate();
    let rx = gpio_d.pd9.into_alternate();

    let serial = dp
        .USART3
        .serial((tx, rx), 921600.bps(), ccdr.peripheral.USART3, &ccdr.clocks)
        .unwrap();

    let (mut tx, _) = serial.split();

    // Initialize DAC, ADC converters.
    let mut dac = DAC::new(spi, dac_cs, (-2.5, 2.5));
    let mut adc = ADC::new(ParallelBus::new(config), convst, eoc, adc_cs, rd, wr);

    dac.init();
    adc.init();

    dac.zero();

    hprintln!("Starting...");

    // Setup RNG.
    let mut rng = XORShift32(SEED);

    // let mu: f32 = 1.5;
    let (a, b, c): (f32, f32, f32) = (0.2, 0.2, 5.7);
    let dt: f32 = 0.05;

    let pulse: f32 = -0.01;
    let span: (f32, f32) = (-5.0, 5.0);

    // Generate the Van der Pol dataset.
    // let dataset = van_der_pol::<STEPS>(mu, dt);
    let dataset = rossler::<STEPS>(a, b, c, dt);

    let recordings = unsafe { &mut RECORDINGS };

    for (t, (x, y, z)) in dataset.enumerate() {
        recordings[t][0] = x;
        recordings[t][1] = y;
        recordings[t][2] = z;

        for (i, point) in [x, z].iter().enumerate() {
            for v in 0..VIRTUAL_NODES {
                dac.all((pulse * point).clamp(-1.0, 1.0));

                delay.delay_us(100u32);

                adc.sample(&mut delay);

                for channel in 0..PHYSICAL_NODES {
                    let index = 4 + (v * PHYSICAL_NODES) + channel;
                    let offset = i * (VIRTUAL_NODES * PHYSICAL_NODES);

                    recordings[t][index + offset] = adc.channel(channel).voltage(span);
                }
            }
        }
    }

    hprintln!("Training readout...");

    dac.zero();

    // Setup the horizon targets.
    for t in 0..(STEPS - HORIZON) {
        recordings[t][3] = recordings[t + HORIZON][1];
    }

    // Establish reservoir performance.
    let mut readout: [f32; STATES] =
        Gaussian::sample(0.0, 0.1, 1000, &mut rng).expect("Unable to initialized readout weights.");
    let mut readout = Vector::from(readout);

    train::<DIMENSION, STATES>(&recordings[..SPLIT], &mut readout, 0.001, false);
    let reservoir_performance =
        test::<DIMENSION, STATES>(&recordings[SPLIT..(STEPS - HORIZON)], &mut readout, false);

    hprintln!("{:?}", reservoir_performance);

    // Establish baseline performance.
    let mut baseline_readout: [f32; 2] =
        Gaussian::sample(0.0, 0.1, 1000, &mut rng).expect("Unable to initialized readout weights.");
    let mut baseline_readout = Vector::from(baseline_readout);

    train::<DIMENSION, 2>(&recordings[..SPLIT], &mut baseline_readout, 0.001, true);
    let baseline_performance = test::<DIMENSION, 2>(
        &recordings[SPLIT..(STEPS - HORIZON)],
        &mut baseline_readout,
        true,
    );

    hprintln!("{:?}", baseline_performance);

    hprintln!("Transmitting data over serial...");

    // Send a synchronization header to align the Python receiver.
    let sync_word: [u8; 4] = [0x53, 0x59, 0x4E, 0x43];
    for byte in sync_word.iter() {
        nb::block!(tx.write(*byte)).ok();
    }

    unsafe {
        let pointer = RECORDINGS.as_ptr() as *const u8;
        let total_bytes = STEPS * DIMENSION * core::mem::size_of::<f32>();
        let byte_slice = core::slice::from_raw_parts(pointer, total_bytes);

        // cortex_m::asm::dmb();

        for byte in byte_slice.iter() {
            nb::block!(tx.write(*byte)).ok();
        }
    }

    hprintln!("Transmission complete.");

    loop {}
}
