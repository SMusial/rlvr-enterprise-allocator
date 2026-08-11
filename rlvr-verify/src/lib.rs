//! rlvr-verify — Mathematical safety invariant assertion harness.
//!
//! Every safety bound from the Master Technical Specification §4
//! is encoded here as a verifiable Rust function.
//!
//! Contract: Ok(()) = invariant holds | Err(String) = violation message.
//!
//! TDD workflow per chapter:
//!   1. Tests below start RED (algorithm not yet implemented)
//!   2. Implement the algorithm in rlvr-core
//!   3. Tests turn GREEN — chapter is done

pub mod invariants;
pub use invariants::*;
