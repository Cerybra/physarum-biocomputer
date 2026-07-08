use embedded_hal::digital::v2::OutputPin;
use embedded_hal::prelude::{
    _embedded_hal_blocking_spi_Write
};

use crate::drivers::utils::voltage_to_code;

pub fn write_word(command: u8, address: u8, code: [u8; 2]) -> [u8; 3] {
    [command | address, code[0], code[1]]
}

#[derive(Copy, Clone, Debug)]
pub enum Command {
    WriteChannel = 0x0,
    UpdateChannel = 0x10,
    WriteChannelUpdateAll = 0x20,
    WriteChannelUpdateChannel = 0x30,
    WriteAll = 0x80,
    UpdateAll = 0x90,
    WriteAllUpdateAll = 0xA0,
}

pub struct DAC<SPI, CS> {
    spi: SPI,

    cs: CS,

    span: (f32, f32),
}

impl<SPI, CS> DAC<SPI, CS>
    where
        SPI: _embedded_hal_blocking_spi_Write<u8>,
        CS: embedded_hal::digital::v2::OutputPin
{
    pub fn new(spi: SPI, cs: CS, span: (f32, f32)) -> DAC<SPI, CS> {
        DAC { spi, cs, span }
    }

    pub fn init(&mut self) -> Result<(), <CS as OutputPin>::Error> {
        self.cs.set_high()
    }

    pub fn select(&mut self) -> Result<(), <CS as OutputPin>::Error> {
        self.cs.set_low()
    }

    pub fn relinquish(&mut self) -> Result<(), <CS as OutputPin>::Error> {
        self.cs.set_high()
    }

    pub fn write(&mut self, channel: u8, voltage: f32) {
        self.spi.write(&write_word(
            Command::WriteChannelUpdateChannel as u8,
            channel,
            voltage_to_code(
                voltage
                    .max(self.span.0)
                    .min(self.span.1),
                self.span
            ).to_be_bytes(),
        ));
    }

    pub fn unroll<const N: usize>(&mut self, voltages: [f32; N]) {
        for channel in 0..8u8 {
            self.write(channel, voltages[channel as usize])
        }
    }

    pub fn multiple<const S: usize>(&mut self, channels: [u8; S], voltages: [f32; S]) {
        for (channel, voltage) in channels.into_iter().zip(voltages.into_iter()) {
            self.complete(channel, voltage)
        }
    }

    pub fn all(&mut self, voltage: f32) {
        for channel in 0..8u8 {
            self.complete(channel, voltage)
        }
    }

    pub fn zero(&mut self) {
        for channel in 0..8u8 {
            self.complete(channel, 0.0)
        }
    }
}

pub trait Complete<T> {

    fn complete(&mut self, channel: T, voltage: f32);
}

impl<SPI, CS> Complete<u8> for DAC<SPI, CS>
    where
        SPI: _embedded_hal_blocking_spi_Write<u8>,
        CS: embedded_hal::digital::v2::OutputPin
{
    fn complete(&mut self, channel: u8, voltage: f32) {
        self.select();
        self.write(channel, voltage);
        self.relinquish();
    }
}

impl<SPI, CS> Complete<usize> for DAC<SPI, CS>
    where
        SPI: _embedded_hal_blocking_spi_Write<u8>,
        CS: embedded_hal::digital::v2::OutputPin
{
    fn complete(&mut self, channel: usize, voltage: f32) {
        self.complete(channel as u8, voltage)
    }
}
