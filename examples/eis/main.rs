#![no_std]
#![no_main]

mod datasets;
mod drivers;
mod linalg;
mod random;

use libm::{exp, fabsf, floorf, sin, sqrtf};

use core::f32::consts::PI;

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

    const FREQUENCIES: usize = 10;
    const BUFFER_SIZE: usize = 500;

    const AMPLIFIER_GAIN: f32 = 100_000_000.0;

    const DC_AMPLITUDE: f32 = 0.0;
    const AC_AMPLITUDE: f32 = 0.05;

    const CYCLES: usize = 5;
    const SAMPLES: usize = 100;

    #[link_section = ".axisram"]
    static mut RECORDINGS: [[f32; BUFFER_SIZE]; FREQUENCIES] = [[0.0; BUFFER_SIZE]; FREQUENCIES];

    #[link_section = ".axisram"]
    static mut EIS_VALUES: [[f32; 4]; FREQUENCIES] = [[0.0; 4]; FREQUENCIES];

    let span: (f32, f32) = (-5.0, 5.0);

    let frequencies: [f32; FREQUENCIES] = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0];

    hprintln!("Pre-biasing sample to {}V DC. Waiting 60 seconds for equilibrium...", DC_AMPLITUDE);

    // Apply the flat DC baseline.
    dac.all(DC_AMPLITUDE.clamp(-1.0, 1.0));

    // Wait for the exponential charging transient to flatten out.
    delay.delay_ms(60_000u32);

    hprintln!("Starting uninterrupted EIS sweep (0.1 Hz to 1000 Hz)...");
    for (i, &frequency) in frequencies.iter().enumerate() {
        let period_us = 1_000_000.0 / frequency;

        // Total raw microsecond calculation.
        let total_delay_us = (period_us / SAMPLES as f32) as u32;

        // Split delay to prevent HAL internal timer overflows.
        let delay_ms = total_delay_us / 1000;
        let delay_us = total_delay_us % 1000;

        let mut index = 0;

        for _ in 0..CYCLES {
            for s in 0..SAMPLES {
                let phase = 2.0 * PI * (s as f32) / (SAMPLES as f32);
                let input = DC_AMPLITUDE + (AC_AMPLITUDE * libm::sinf(phase));

                dac.all(input.clamp(-1.0, 1.0));

                // Execute safe compound hardware delay.
                if delay_ms > 0 {
                    delay.delay_ms(delay_ms);
                }
                if delay_us > 0 {
                    delay.delay_us(delay_us);
                }

                adc.sample(&mut delay);
                let voltage = adc.channel(0).voltage(span);

                // Store true physical current (Amperes) directly into memory.
                unsafe {
                    RECORDINGS[i][index] = voltage / AMPLIFIER_GAIN;
                }

                index += 1;
            }
        }
    }

    dac.zero();

    hprintln!("Acquisition complete. Processing data arrays via Lock-In...");
    for (i, &frequency) in frequencies.iter().enumerate() {
        let discard_cycles = 1;
        let total_samples = CYCLES * SAMPLES;
        let starting_index = discard_cycles * SAMPLES;

        let (mut sum_i_sin, mut sum_i_cos) = (0.0, 0.0);
        let mut valid_samples = 0;

        for index in starting_index..total_samples {
            let s = index % SAMPLES;
            let phase = 2.0 * PI * (s as f32) / (SAMPLES as f32);

            // Read safely from our massive static buffer.
            let current_val = unsafe {
                RECORDINGS[i][index]
            };

            sum_i_sin += current_val * libm::sinf(phase);
            sum_i_cos += current_val * libm::cosf(phase);

            valid_samples += 1;
        }

        // Hardware Compensation Constants (10pF || 10M corner pole).
        let f_corner = 1591.5494;
        let w_ratio = frequency / f_corner;

        let i_real_raw = (sum_i_sin / valid_samples as f32) * 2.0;
        let i_imag_raw = (sum_i_cos / valid_samples as f32) * 2.0;

        let i_real = i_real_raw - (i_imag_raw * w_ratio);
        let i_imag = i_imag_raw + (i_real_raw * w_ratio);

        let denominator = (i_real * i_real) + (i_imag * i_imag);

        let z_real = (AC_AMPLITUDE * i_real) / denominator;
        let z_imag = -(AC_AMPLITUDE * i_imag) / denominator;

        let magnitude = libm::sqrtf((z_real * z_real) + (z_imag * z_imag));
        let phase_angle = libm::atan2f(z_imag, z_real) * (180.0 / PI);

        // Populate results back to static memory allocation.
        unsafe {
            EIS_VALUES[i][0] = z_real;
            EIS_VALUES[i][1] = z_imag;
            EIS_VALUES[i][2] = magnitude;
            EIS_VALUES[i][3] = phase_angle;
        }
    }

    hprintln!("EIS Sweep complete...");
    hprintln!("FrequencyHz,ZRealOhms,ZImagOhms,MagnitudeOhms,PhaseDegree");
    for i in 0..FREQUENCIES {
        unsafe {
            hprintln!("{},{},{},{},{}",
            frequencies[i],
            EIS_VALUES[i][0],
            EIS_VALUES[i][1],
            EIS_VALUES[i][2],
            EIS_VALUES[i][3]
        );
        }
    }

    loop {}
}
