#![no_std]
#![no_main]

mod random;
mod drivers;
mod linalg;
mod datasets;

use core::f64::consts::PI;
use libm::{exp, floorf, sin, sqrtf, fabsf};

use panic_halt as _;

use cortex_m_semihosting::hprintln;
use embedded_hal::blocking::spi::WriteIter;

use stm32h7xx_hal::{prelude::*, serial::Error, spi};

use stm32h7xx_hal::rng::RngCore;

use crate::drivers::adc::{ParallelBus, ParallelBusConfig, ADC};
use crate::drivers::dac::DAC;
use crate::random::generator::Generator;

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

    // Initialize DAC, ADC converters.
    let mut dac = DAC::new(spi, dac_cs, (-2.5, 2.5));
    let mut adc = ADC::new(ParallelBus::new(config), convst, eoc, adc_cs, rd, wr);

    dac.init();
    adc.init();

    dac.zero();

    hprintln!("Starting Sweep...");

    let amplitude: f32 = 0.5;
    let span: (f32, f32) = (-5.0, 5.0);

    let mut sweep: [(f32, [f32; 8]); 360 * 5] = [(0.0, [0.0; 8]); 360 * 5];

    for cycle in 0..5 {
        for t in 0..360 {
            let signal = amplitude * sin(((t as f64) * PI * (1.0 / 180.0))) as f32;

            dac.complete(0, signal.clamp(-1.0, 1.0));
            dac.complete(1, signal.clamp(-1.0, 1.0));
            dac.complete(2, signal.clamp(-1.0, 1.0));
            dac.complete(3, signal.clamp(-1.0, 1.0));
            dac.complete(4, signal.clamp(-1.0, 1.0));
            dac.complete(5, signal.clamp(-1.0, 1.0));
            dac.complete(6, signal.clamp(-1.0, 1.0));
            dac.complete(7, signal.clamp(-1.0, 1.0));

            adc.sample(&mut delay);

            let measurements = [
                adc.channel(0).voltage(span),
                adc.channel(1).voltage(span),
                adc.channel(2).voltage(span),
                adc.channel(3).voltage(span),
                adc.channel(4).voltage(span),
                adc.channel(5).voltage(span),
                adc.channel(6).voltage(span),
                adc.channel(7).voltage(span)
            ];

            sweep[t + (360 * cycle)] = (signal, measurements);

            delay.delay_ms(1000u32);
        }
    }

    dac.zero();

    hprintln!("{:?}", sweep);

    loop {}
}
