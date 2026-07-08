use embedded_hal::blocking::i2c::Write;

pub type RowSelector<I2C> = SelectorInterface<I2C>;
pub type ColumnSelector<I2C> = SelectorInterface<I2C>;

pub struct Selector<I2C> {
    address: u8,

    i2c: I2C
}

impl<I2C> Selector<I2C>
    where
        I2C: Write
{
    pub fn new(address: u8, i2c: I2C) -> Selector<I2C> {
        Selector { address, i2c }
    }

    pub fn select(&mut self, selections: u8) {
        self.i2c.write(self.address, &[0, selections]);
    }
}

pub struct SelectorInterface<I2C> {
    selector: Selector<I2C>,

    state: u8,
}

impl<I2C> SelectorInterface<I2C>
    where
        I2C: Write
{
    pub fn with(selector: Selector<I2C>) -> SelectorInterface<I2C> {
        SelectorInterface { selector, state: 0 }
    }

    pub fn select(&mut self, index: u8) {
        self.selector.select(1 << index);
    }

    pub fn all(&mut self) {
        self.selector.select(u8::MAX);
    }

    pub fn none(&mut self) {
        self.selector.select(0);
    }

    pub fn reset(&mut self) {
        self.state = 0;
        self.none();
    }
}

pub trait Persist<T> {

    fn persist(&mut self, index: T);
}

impl<I2C> Persist<u8> for SelectorInterface<I2C>
    where
        I2C: Write
{
    fn persist(&mut self, index: u8) {
        self.state |= (1 << index);
        self.selector.select(self.state);
    }
}

impl<I2C> Persist<usize> for SelectorInterface<I2C>
    where
        I2C: Write {
    fn persist(&mut self, index: usize) {
        self.persist(index as u8);
    }
}
