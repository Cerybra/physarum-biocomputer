#![no_std]
#![no_main]

mod random;
mod drivers;
mod linalg;
mod datasets;

use core::fmt::Write;
use core::f64::consts::PI;

use libm::{exp, floorf, sin, sqrtf, fabsf};

use panic_halt as _;

use cortex_m_semihosting::hprintln;

use embedded_hal::blocking::spi::WriteIter;
use embedded_hal::serial::{Read as ReadSerial, Write as WriteSerial};

use stm32h7xx_hal::{prelude::*, serial::Error, spi};

use stm32h7xx_hal::rng::RngCore;

use crate::drivers::adc::{ParallelBus, ParallelBusConfig, ADC};
use crate::drivers::dac::DAC;

use crate::linalg::{Vector, Array, Dot};

use crate::random::generator::{Generator, XORShift32};
use crate::random::distribution::Gaussian;

#[cortex_m_rt::entry]
fn main() -> ! {
    hprintln!("Initialized...");

    // Get access to the core peripherals
    let core = cortex_m::Peripherals::take().unwrap();

    // Get access to the device specific peripherals from the peripheral access crate
    let dp = stm32h7xx_hal::stm32::Peripherals::take().unwrap();

    // Take ownership over the RCC devices and convert them into the corresponding HAL structs
    let rcc = dp.RCC.constrain();

    let pwr = dp.PWR.constrain();
    let pwrcfg = pwr.freeze();

    // Freeze the configuration of all the clocks in the system and
    // retrieve the Core Clock Distribution and Reset (CCDR) object
    let rcc = rcc.sys_ck(400.MHz()).use_hse(8.MHz()).bypass_hse();
    let ccdr = rcc.freeze(pwrcfg, &dp.SYSCFG);

    let mut delay = core.SYST.delay(ccdr.clocks);

    let mut rng = dp.RNG.constrain(ccdr.peripheral.RNG, &ccdr.clocks);

    // Acquire the GPIO peripherals
    let gpio_a = dp.GPIOA.split(ccdr.peripheral.GPIOA);
    let gpio_b = dp.GPIOB.split(ccdr.peripheral.GPIOB);
    let gpio_d = dp.GPIOD.split(ccdr.peripheral.GPIOD);
    let gpio_e = dp.GPIOE.split(ccdr.peripheral.GPIOE);
    let gpio_f = dp.GPIOF.split(ccdr.peripheral.GPIOF);

    // Configuring the pins for the ADC parallel interface
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

    // Initialize required CS pins
    let dac_cs = gpio_a.pa0.into_push_pull_output();
    let adc_cs = gpio_d.pd5.into_push_pull_output();

    hprintln!("Initializing SPI ... ");

    // Initialize the SPI peripheral
    let mut spi = dp.SPI5.spi(
        (sck, miso, mosi),
        spi::MODE_0,
        3.MHz(),
        ccdr.peripheral.SPI5,
        &ccdr.clocks,
    );

    // Initialize DAC, ADC converters
    let mut dac = DAC::new(spi, dac_cs, (-2.5, 2.5));
    let mut adc = ADC::new(ParallelBus::new(config), convst, eoc, adc_cs, rd, wr);

    // Initialize serial
    let tx = gpio_d.pd8.into_alternate();
    let rx = gpio_d.pd9.into_alternate();

    let serial = dp
        .USART3
        .serial((tx, rx), 115200.bps(), ccdr.peripheral.USART3, &ccdr.clocks)
        .unwrap();

    let (mut tx, _) = serial.split();

    dac.init();
    adc.init();

    dac.zero();

    hprintln!("Starting...");

    const SEED: u32 = 237643274;

    const EPOCHS: usize = 1;
    const STEPS: usize = 850;

    const PHYSICAL_NODES: usize = 8;
    const VIRTUAL_NODES: usize = 4;

    const DIMENSION: usize = (PHYSICAL_NODES * VIRTUAL_NODES) + 2;

    // Setup RNG.
    let mut rng = XORShift32(SEED);

    // The last 32 are reservoir recordings, while the first two are the point and target, respectively.
    let mut recordings: [[f32; DIMENSION]; EPOCHS * STEPS] = [[0.0; DIMENSION]; EPOCHS * STEPS];

    let pulse: f32 = -0.1;
    let span: (f32, f32) = (-5.0, 5.0);

    let mut iteration: usize = 0;

    for _ in 0..EPOCHS {
        for (point, target) in temporal_xor_iter(STEPS) {
            recordings[iteration][0] = point;
            recordings[iteration][1] = target;

            for v in 0..VIRTUAL_NODES {
                dac.all((pulse * point).clamp(-1.0, 1.0));

                delay.delay_us(10u32);

                adc.sample(&mut delay);

                for channel in 0..PHYSICAL_NODES {
                    let index = 2 + (v * PHYSICAL_NODES) + channel;

                    recordings[iteration][index] = adc.channel(channel).voltage(span);
                }
            }

            dac.zero();

            delay.delay_ms(10u32);

            iteration += 1;
        }
    }

    dac.zero();

    // Establish reservoir performance.
    let mut readout: [f32; 32] = Gaussian::sample(0.0, 0.1, 1000, &mut rng)
        .expect("Unable to initialized readout weights.");

    let mut readout = Vector::from(readout);

    train::<DIMENSION, { DIMENSION - 2 }>(&recordings, &mut readout, 0.001, false);
    let reservoir_performance = test::<DIMENSION, { DIMENSION - 2 }>(&recordings, &mut readout, false);

    // hprintln!("{:?}", readout.array());
    hprintln!("{:?}", reservoir_performance);

    // Establish baseline performance.
    let mut readout: [f32; 32] = Gaussian::sample(0.0, 0.1, 1000, &mut rng)
        .expect("Unable to initialized readout weights.");

    let mut readout = Vector::from(readout);

    train::<DIMENSION, { DIMENSION - 2 }>(&recordings, &mut readout, 0.001, true);
    let baseline_performance = test::<DIMENSION, { DIMENSION - 2 }>(&recordings, &mut readout, true);

    // hprintln!("{:?}", readout.array());
    hprintln!("{:?}", baseline_performance);

    hprintln!("{:?}", recordings);

    loop {}
}

fn temporal_xor_iter(timesteps: usize) -> impl Iterator<Item = (f32, f32)> {
    let pattern = [0, 0, 1, 1, 0, 1];

    (0..timesteps).scan(None, move |prev_x, i| {
        let x = pattern[i % pattern.len()];

        let y = match *prev_x {
            None => 0,
            Some(p) => x ^ p
        };

        *prev_x = Some(x);
        Some((x as f32, y as f32))
    })
}

fn train<const S: usize, const N: usize>(
    recordings: &[[f32; S]],
    readout: &mut Vector<N>,
    lr: f32,
    baseline: bool
) {
    for element in recordings.iter() {
        let (point, target) = (element[0], element[1]);

        let state = if baseline {
            Vector::full(point)
        } else {
            Vector::from(core::array::from_fn(|i| element[i + 2]))
        };

        let error = target - state.dot(&mut *readout);

        *readout = *readout + (state * (lr * error));
    }
}

fn test<const S: usize, const N: usize>(
    recordings: &[[f32; S]],
    readout: &Vector<N>,
    baseline: bool
) -> f32 {
    let correct_count = recordings.iter()
        .filter(|row| {
            let (point, target) = (row[0], row[1]);

            let state = if baseline {
                Vector::full(point)
            } else {
                Vector::from(core::array::from_fn(|i| row[i + 2]))
            };

            let prediction = state.dot(readout);
            let output = if prediction > 0.5 { 1.0 } else { 0.0 };

            (output - target).abs() < f32::EPSILON
        })
        .count();

    (correct_count as f32 / recordings.len() as f32) * 100.0
}
