use std::collections::VecDeque;
use crate::events::TelemetryEvent;

pub struct RingBuffer {
    cap:  usize,
    data: VecDeque<TelemetryEvent>,
}

impl RingBuffer {
    pub fn new(cap: usize) -> Self { Self { cap, data: VecDeque::with_capacity(cap) } }
    pub fn emit(&mut self, e: TelemetryEvent) {
        if self.data.len() == self.cap { self.data.pop_front(); }
        self.data.push_back(e);
    }
    pub fn drain(&mut self) -> Vec<TelemetryEvent> { self.data.drain(..).collect() }
    pub fn len(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_buffer_capacity() {
        let mut rb = RingBuffer::new(3);
        for i in 0..5 { rb.emit(TelemetryEvent::Reward { step: i, value: i as f64 }); }
        assert_eq!(rb.len(), 3);
    }
    #[test]
    fn drain_empties_buffer() {
        let mut rb = RingBuffer::new(10);
        rb.emit(TelemetryEvent::Epsilon { episode: 0, value: 1.0 });
        let drained = rb.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(rb.len(), 0);
    }
}
