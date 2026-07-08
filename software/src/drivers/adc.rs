use stm32h7xx_hal::delay::Delay;

use stm32h7xx_hal::gpio::ErasedPin;
use stm32h7xx_hal::gpio::{
    Input,
    Output,
    PushPull,
    Analog
};

use embedded_hal::digital::v2::{InputPin, OutputPin};
use embedded_hal::blocking::delay::DelayUs;

use crate::drivers::utils::code_to_voltage;

pub struct ParallelBusConfig<
    DB0,
    DB1,
    DB2,
    DB3,
    DB4,
    DB5,
    DB6,
    DB7,
    DB8,
    DB9,
    DB10,
    DB11,
    DB12,
    DB13,
    DB14,
    DB15,
> {
    pub db0: DB0,
    pub db1: DB1,
    pub db2: DB2,
    pub db3: DB3,
    pub db4: DB4,
    pub db5: DB5,
    pub db6: DB6,
    pub db7: DB7,
    pub db8: DB8,
    pub db9: DB9,
    pub db10: DB10,
    pub db11: DB11,
    pub db12: DB12,
    pub db13: DB13,
    pub db14: DB14,
    pub db15: DB15,
}

pub struct ParallelBus {
    bits: [stm32h7xx_hal::gpio::ErasedPin<Input>; 16],
}

impl ParallelBus {
    pub fn new<
        DB0,
        DB1,
        DB2,
        DB3,
        DB4,
        DB5,
        DB6,
        DB7,
        DB8,
        DB9,
        DB10,
        DB11,
        DB12,
        DB13,
        DB14,
        DB15,
    >(
        config: ParallelBusConfig<
            DB0,
            DB1,
            DB2,
            DB3,
            DB4,
            DB5,
            DB6,
            DB7,
            DB8,
            DB9,
            DB10,
            DB11,
            DB12,
            DB13,
            DB14,
            DB15,
        >,
    ) -> ParallelBus
        where
            DB0: ErasePin<PinMode = Input>,
            DB1: ErasePin<PinMode = Input>,
            DB2: ErasePin<PinMode = Input>,
            DB3: ErasePin<PinMode = Input>,
            DB4: ErasePin<PinMode = Input>,
            DB5: ErasePin<PinMode = Input>,
            DB6: ErasePin<PinMode = Input>,
            DB7: ErasePin<PinMode = Input>,
            DB8: ErasePin<PinMode = Input>,
            DB9: ErasePin<PinMode = Input>,
            DB10: ErasePin<PinMode = Input>,
            DB11: ErasePin<PinMode = Input>,
            DB12: ErasePin<PinMode = Input>,
            DB13: ErasePin<PinMode = Input>,
            DB14: ErasePin<PinMode = Input>,
            DB15: ErasePin<PinMode = Input>,
    {
        ParallelBus {
            bits: [
                config.db0.erase(),
                config.db1.erase(),
                config.db2.erase(),
                config.db3.erase(),
                config.db4.erase(),
                config.db5.erase(),
                config.db6.erase(),
                config.db7.erase(),
                config.db8.erase(),
                config.db9.erase(),
                config.db10.erase(),
                config.db11.erase(),
                config.db12.erase(),
                config.db13.erase(),
                config.db14.erase(),
                config.db15.erase(),
            ],
        }
    }

    pub fn expose(
        &self,
        bit: usize,
    ) -> &stm32h7xx_hal::gpio::ErasedPin<Input> {
        &self.bits[bit]
    }
}

trait ErasePin {
    type PinMode;

    fn erase(self) -> stm32h7xx_hal::gpio::ErasedPin<Self::PinMode>;
}

impl<const P: char, const N: u8> ErasePin for stm32h7xx_hal::gpio::Pin<P, N, Input>
{
    type PinMode = Input;

    fn erase(self) -> ErasedPin<Self::PinMode> {
        self.erase()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Channel(u16);

impl Channel {
    pub fn code(self) -> u16 {
        self.0
    }

    pub fn voltage(self, span: (f32, f32)) -> f32 {
        code_to_voltage(self.0, span)
    }
}

impl Default for Channel {
    fn default() -> Self {
        Channel(0)
    }
}

pub struct ADC<CONVST, EOC, CS, RD, WR> {
    bus: ParallelBus,

    convst: CONVST,
    eoc: EOC,

    cs: CS,
    rd: RD,

    wr: WR,

    channels: [Channel; 8],
}

impl<CONVST, EOC, CS, RD, WR> ADC<CONVST, EOC, CS, RD, WR>
    where
        CONVST: OutputPin,
        EOC: InputPin,
        CS: OutputPin,
        RD: OutputPin,
        WR: OutputPin
{
    pub fn new(
        bus: ParallelBus,
        convst: CONVST,
        eoc: EOC,
        cs: CS,
        rd: RD,
        wr: WR,
    ) -> ADC<CONVST, EOC, CS, RD, WR> {
        ADC {
            bus,

            convst,
            eoc,

            cs,
            rd,

            wr,

            channels: [Default::default(); 8]
        }
    }

    pub fn init(&mut self) {
        self.convst.set_high();

        self.cs.set_high();
        self.rd.set_high();
        self.wr.set_high();
    }

    pub fn convert(&mut self, delay: &mut Delay) {
        self.convst.set_low();

        delay.delay_us(1u32);

        self.convst.set_high();
    }

    pub fn read(&mut self) -> u16 {
        let mut code = 0;

        for (n, bit) in self.bus.bits.iter().enumerate() {
            if bit.is_high() {
                code = code | (1 << n);
            }
        }

        code
    }

    pub fn sample(&mut self, delay: &mut Delay) {
        self.convert(delay);

        // Wait for the conversion to complete.
        while let Ok(state) = self.eoc.is_high() {
            if !state {
                break
            }
        }

        self.convst.set_low();
        self.cs.set_low();

        for i in 0..8usize {
            self.rd.set_low();

            self.channels[i] = Channel(self.read());

            self.rd.set_high();
        }

        self.rd.set_high();
        self.cs.set_high();

        self.convst.set_high();
    }

    pub fn channel(&self, channel: usize) -> &Channel {
        &self.channels[channel]
    }

    pub fn channels(&self) -> &[Channel; 8] {
        &self.channels
    }
}
