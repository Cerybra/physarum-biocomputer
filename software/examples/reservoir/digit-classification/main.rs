#![no_std]
#![no_main]

mod datasets;
mod drivers;
mod linalg;
mod random;

use core::fmt::Write;

use panic_halt as _;

use cortex_m_semihosting::hprintln;

use embedded_hal::blocking::spi::WriteIter;
use embedded_hal::serial::{Read as ReadSerial, Write as WriteSerial};

use stm32h7xx_hal::{nb, prelude::*, serial::Error, spi};

use crate::drivers::adc::{ParallelBus, ParallelBusConfig, ADC};
use crate::drivers::dac::DAC;

// Command signals for the array exchange protocol.
const STRT: [u8; 4] = *b"STRT";
const RQST: [u8; 4] = *b"RQST";
const RCVD: [u8; 4] = *b"RCVD";
const NACK: [u8; 4] = *b"NACK";
const FNSH: [u8; 4] = *b"FNSH";

// Predetermined sizes of the arrays.
const ROWS: usize = 8;
const COLS: usize = 8;

fn read_exact<Rx: ReadSerial<u8>>(rx: &mut Rx, buf: &mut [u8]) -> Result<(), ()> {
    for byte in buf.iter_mut() {
        *byte = nb::block!(rx.read()).map_err(|_| ())?;
    }

    Ok(())
}

fn write_all<Tx: WriteSerial<u8>>(tx: &mut Tx, buf: &[u8]) -> Result<(), ()> {
    for &byte in buf {
        nb::block!(tx.write(byte)).map_err(|_| ())?;
    }

    Ok(())
}

fn checksum_bytes(data: &[u8]) -> u32 {
    data.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32))
}

fn wait_for_marker<Rx: ReadSerial<u8>>(rx: &mut Rx, marker: &[u8]) -> Result<(), ()> {
    let mut buffer = [0u8; 4];
    let mut index = 0;

    loop {
        buffer[index % 4] = nb::block!(rx.read()).map_err(|_| ())?;

        index += 1;

        if &buffer == marker {
            return Ok(());
        }
    }
}

fn receive_array<Rx: ReadSerial<u8>, Tx: WriteSerial<u8>, const N: usize, const M: usize>(
    rx: &mut Rx,
    tx: &mut Tx,
    array_buf: &mut [u8],
    payload_buf: &mut [u8],
) -> Result<[[f32; M]; N], ()> {
    wait_for_marker(rx, &STRT)?;

    let mut header = [0u8; 8];
    let mut checksum = [0u8; 4];

    read_exact(rx, &mut header)?;

    let rows = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    if rows != N || cols != M {
        write_all(tx, &NACK)?;

        return Err(());
    } else {
        read_exact(rx, array_buf)?;
        read_exact(rx, &mut checksum)?;

        let received_checksum = u32::from_le_bytes(checksum);

        payload_buf[..4].copy_from_slice(&STRT);
        payload_buf[4..12].copy_from_slice(&header);
        payload_buf[12..].copy_from_slice(&array_buf[..]);

        let computed_checksum = checksum_bytes(&payload_buf);

        if received_checksum != computed_checksum {
            write_all(tx, &NACK)?;

            return Err(());
        } else {
            let mut array = [[0.0f32; M]; N];

            for i in 0..(N * M) {
                let bytes: [u8; 4] = array_buf[(i * 4)..((i + 1) * 4)].try_into().unwrap();

                array[i / M][i % M] = f32::from_le_bytes(bytes);
            }

            write_all(tx, &RCVD)?;

            Ok(array)
        }
    }
}

fn send_array<Tx: WriteSerial<u8>>(tx: &mut Tx, array: &[[f32; 8]; 8]) -> Result<(), ()> {
    // Prepare header
    let rows = 8u32.to_le_bytes();
    let cols = 8u32.to_le_bytes();

    // Prepare payload: marker (4) + header (8) + data (8*8*4) = 4 + 8 + 256 = 268 bytes
    let mut payload = [0u8; 268];

    payload[..4].copy_from_slice(&STRT);
    payload[4..8].copy_from_slice(&rows);
    payload[8..12].copy_from_slice(&cols);

    // Copy array data as bytes (row-major order)
    for i in 0..8 {
        for j in 0..8 {
            let offset = 12 + (i * 8 + j) * 4;
            payload[offset..offset + 4].copy_from_slice(&array[i][j].to_le_bytes());
        }
    }

    // Compute checksum over payload
    let checksum = checksum_bytes(&payload).to_le_bytes();

    // Final message: payload (268) + checksum (4) = 272 bytes
    let mut message = [0u8; 272];

    message[..268].copy_from_slice(&payload);
    message[268..].copy_from_slice(&checksum);

    write_all(tx, &message)
}

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

    // Setup the use of delays.
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

    // Initialize serial.
    let tx = gpio_d.pd8.into_alternate();
    let rx = gpio_d.pd9.into_alternate();

    // Configure serial.
    let serial = dp
        .USART3
        .serial((tx, rx), 115200.bps(), ccdr.peripheral.USART3, &ccdr.clocks)
        .unwrap();

    let (mut tx, mut rx) = serial.split();

    dac.init();
    adc.init();

    dac.zero();

    hprintln!("Starting...");

    let amplitude: f32 = 0.1;
    let pulse: f32 = 0.01;

    let span: (f32, f32) = (-5.0, 5.0);

    let mut array_buf = [0u8; ROWS * COLS * 4];
    let mut payload_buf = [0u8; 4 + 8 + ROWS * COLS * 4];

    let mut signal_buf = [0u8; 4];

    loop {
        // Request new image from host
        write_all(&mut tx, &RQST);

        // Wait for either FNSH or STRT marker.
        read_exact(&mut rx, &mut signal_buf);

        if &signal_buf == &FNSH {
            // Host is finished, acknowledge and break.
            write_all(&mut tx, &RCVD);
            break;
        }

        if &signal_buf == &STRT {
            if let Ok(array) = receive_array::<_, _, 8, 8>(&mut rx, &mut tx, &mut array_buf, &mut payload_buf) {
                let mut recordings = [[0.0f32; COLS]; ROWS];

                for t in 0..COLS {
                    dac.complete(0, (array[0][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(1, (array[1][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(2, (array[2][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(3, (array[3][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(4, (array[4][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(5, (array[5][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(6, (array[6][t] * amplitude).clamp(-1.0, 1.0));
                    dac.complete(7, (array[7][t] * amplitude).clamp(-1.0, 1.0));

                    delay.delay_ms(20u32);
                }

                dac.complete(0, pulse.clamp(-1.0, 1.0));
                dac.complete(1, pulse.clamp(-1.0, 1.0));
                dac.complete(2, pulse.clamp(-1.0, 1.0));
                dac.complete(3, pulse.clamp(-1.0, 1.0));
                dac.complete(4, pulse.clamp(-1.0, 1.0));
                dac.complete(5, pulse.clamp(-1.0, 1.0));
                dac.complete(6, pulse.clamp(-1.0, 1.0));
                dac.complete(7, pulse.clamp(-1.0, 1.0));

                adc.sample(&mut delay);

                dac.zero();

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

                recordings[0] = measurements;

                send_array(&mut tx, &recordings);

                delay.delay_ms(100u32);
            }
        } else {
            // Unexpected signal, optionally handle error or resync.
        }

        dac.zero();
    }

    dac.zero();

    loop {}
}
