#![no_std]
#![no_main]

mod random;
mod drivers;
mod linalg;
mod datasets;

use libm::{exp, fabsf, floorf, sin, sqrtf};

use panic_halt as _;

use cortex_m_semihosting::hprintln;
use embedded_hal::blocking::spi::WriteIter;

use stm32h7xx_hal::{prelude::*, serial::Error, spi};

use stm32h7xx_hal::rng::RngCore;

use crate::drivers::adc::{ParallelBus, ParallelBusConfig, ADC};
use crate::drivers::dac::{DAC, Complete};

use crate::drivers::selector::{Selector, SelectorInterface, Persist};
use crate::random::distribution::Gaussian;

use crate::random::generator::{Generator, XORShift32};

#[cortex_m_rt::entry]
fn main() -> ! {
    hprintln!("Initialized...");

    // Get access to the core peripherals.
    let core = cortex_m::Peripherals::take().unwrap();

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

    // Configuring I2C pins.
    let scl1 = gpio_f.pf1.into_alternate_open_drain();
    let sda1 = gpio_f.pf0.into_alternate_open_drain();

    // Initialize the I2C peripheral.
    let mut i2c1 = dp
        .I2C2
        .i2c((scl1, sda1), 100.kHz(), ccdr.peripheral.I2C2, &ccdr.clocks);

    // Initialize DAC, ADC converters.
    let mut dac = DAC::new(spi, dac_cs, (-2.5, 2.5));
    let mut adc = ADC::new(ParallelBus::new(config), convst, eoc, adc_cs, rd, wr);

    // Initialize the selector.
    let mut selector = SelectorInterface::with(
        Selector::new(0x4C, i2c1)
    );

    dac.init();
    adc.init();

    dac.zero();

    const SEED: u32 = 237643274;

    const EPOCHS: usize = 5;
    const SAMPLES: usize = 100;

    let lr: f32 = -1.0;

    let pulse: f32 = 0.01;
    let span: (f32, f32) = (-5.0, 5.0);

    let slope: f32 = 0.88;

    // Setup RNG.
    let mut rng = XORShift32(SEED);

    // The first two are the input and target, respectively, while the rest are
    // predictions and updates, also in that order.
    let mut recordings: [[f32; 18]; EPOCHS * SAMPLES] = [[0.0; 18]; EPOCHS * SAMPLES];

    let inputs: [f32; 100] = core::array::from_fn(|i| (i as f32) / 100.0);
    let noise: [f32; 100] = Gaussian::sample(0.0, 0.01, 10000, &mut rng)
        .expect("Unable to generate dataset noise component.");

    let targets: [f32; 100] = core::array::from_fn(|i| (slope * inputs[i]) + noise[i] );

    hprintln!(
        "Performing regression task with the following parameters:\t\
        SLOPE: {}\n\t\
        PULSE: {}\n\t\
        LR: {}\n\t\
        SAMPLES: {}\n\t",
        slope,
        pulse,
        lr,
        SAMPLES
    );

    hprintln!("Starting...");

    for epoch in 0..EPOCHS {
        for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
            let index = (epoch * SAMPLES) + i;

            recordings[index][0] = *input;
            recordings[index][1] = *target;

            selector.reset();
            dac.all(input.clamp(-1.0, 1.0));

            // Duration of the readout pulse.
            delay.delay_us(10u32);

            adc.sample(&mut delay);
            dac.zero();

            for channel in 0..8 {
                let prediction = adc.channel(channel).voltage(span);
                let update = (lr * input * (prediction - target)).clamp(-0.1, 0.1);

                selector.persist(channel);

                dac.complete(channel, update);

                recordings[index][2 + channel] = prediction;
                recordings[index][2 + 8 + channel] = update;
            }

            // Duration of the gradient updates.
            delay.delay_us(100u32);

            dac.zero();

            // Pulling the devices to ground between data points.
            selector.all();
            delay.delay_ms(1u32);

            selector.reset();
            selector.none();

            dac.zero();
        }
    }

    hprintln!("{:?}", recordings);

    dac.zero();

    loop {}
}
