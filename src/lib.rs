#![warn(missing_debug_implementations, rust_2018_idioms, clippy::pedantic)]
//!
//! The Rank-Biased Centroids (RBC) rank fusion method to combine multiple-rankings of objects.
//!
//! This code implements the RBC rank fusion method, as described in:
//!
//!```bibtex
//! @inproceedings{DBLP:conf/sigir/BaileyMST17,
//!    author    = {Peter Bailey and
//!                 Alistair Moffat and
//!                 Falk Scholer and
//!                 Paul Thomas},
//!   title     = {Retrieval Consistency in the Presence of Query Variations},
//!   booktitle = {Proceedings of the 40th International {ACM} {SIGIR} Conference on
//!                Research and Development in Information Retrieval, Shinjuku, Tokyo,
//!                Japan, August 7-11, 2017},
//!   pages     = {395--404},
//!   publisher = {{ACM}},
//!   year      = {2017},
//!   url       = {https://doi.org/10.1145/3077136.3080839},
//!   doi       = {10.1145/3077136.3080839},
//!   timestamp = {Wed, 25 Sep 2019 16:43:14 +0200},
//!   biburl    = {https://dblp.org/rec/conf/sigir/BaileyMST17.bib},
//! }
//!```
//!
//! The fundamental step in the working of RBC is the usage a `persistence` parameter (`p` or `phi`) to to fusion multiple ranked lists based only on rank information. Larger values of `p` give higher importance to elements at the top of each ranking. From the paper:
//!
//! > As extreme values, consider `p = 0` and `p = 1`. When `p = 0`, the agents only ever examine the first item in each of the input rankings, and the fused output is by decreasing score of firstrst preference; this is somewhat akin to a first-past-the-post election regime. When `p = 1`, each agent examines the whole of every list, and the fused ordering is determined by the number of lists that contain each item – a kind of "popularity count" of each item across the input sets. In between these extremes, the expected depth reached by the agents viewing the rankings is given by `1/(1 − p)`. For example, when `p = 0.9`, on average the first 10 items in each ranking are being used to contribute to the fused ordering; of course, in aggregate, across the whole universe of agents, all of the items in every ranking contribute to the overall outcome.
//!
//! More from the paper:
//!
//! > Each item at rank 1 <= x <= n when the rankings are over n items, we suggest that a geometrically decaying weight function be employed, with the distribution of d over depths x given by (1 − p) p^{x-1} for some value 0 <= p <= 1 determined by considering the purpose for which the fused ranking is being constructed.
//!
//! # Example fusion
//!
//! For example (also taken from the paper) for diffent rank orderings (R1-R4) of items A-G:
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th>Rank</th>
//!             <th>R1</th>
//!             <th>R2</th>
//!             <th>R3</th>
//!             <th>R4</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>1</td>
//!             <td>A</td>
//!             <td>B</td>
//!             <td>A</td>
//!             <td>G</td>
//!         </td>
//!         <tr>
//!             <td>2</td>
//!             <td>D</td>
//!             <td>D</td>
//!             <td>B</td>
//!             <td>D</td>
//!         </tr>
//!         <tr>
//!             <td>3</td>
//!             <td>B</td>
//!             <td>E</td>
//!             <td>D</td>
//!             <td>E</td>
//!         </tr>
//!         <tr>
//!             <td>4</td>
//!             <td>C</td>
//!             <td>C</td>
//!             <td>C</td>
//!             <td>A</td>
//!         </tr>
//!         <tr>
//!             <td>5</td>
//!             <td>G</td>
//!             <td>-</td>
//!             <td>G</td>
//!             <td>F</td>
//!         </tr>
//!         <tr>
//!             <td>6</td>
//!             <td>F</td>
//!             <td>-</td>
//!             <td>F</td>
//!             <td>C</td>
//!         </tr>
//!         <tr>
//!             <td>7</td>
//!             <td>-</td>
//!             <td>-</td>
//!             <td>E</td>
//!             <td>-</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! Depending on the persistence parameter `p` will result in different output orderings based on each items accumulated weights:
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th>Rank</th>
//!             <th>p=0.6</th>
//!             <th>p=0.8</th>
//!             <th>p=0.9</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>1</td>
//!             <td>A (0.89)</td>
//!             <td>D (0.61)</td>
//!             <td>D (0.35)</td>
//!         </td>
//!         <tr>
//!             <td>2</td>
//!             <td>D (0.86)</td>
//!             <td>A (0.50)</td>
//!             <td>D (0.35)</td>
//!         </tr>
//!         <tr>
//!             <td>3</td>
//!             <td>B (0.78)</td>
//!             <td>B (0.49)</td>
//!             <td>A (0.27)</td>
//!         </tr>
//!         <tr>
//!             <td>4</td>
//!             <td>G (0.50) </td>
//!             <td>C (0.37)</td>
//!             <td>B (0.27)</td>
//!         </tr>
//!         <tr>
//!             <td>5</td>
//!             <td>E (0.31)</td>
//!             <td>G (0.37)</td>
//!             <td>G (0.23)</td>
//!         </tr>
//!         <tr>
//!             <td>6</td>
//!             <td>C (0.29)</td>
//!             <td>E (0.31)</td>
//!             <td>E (0.22)</td>
//!         </tr>
//!         <tr>
//!             <td>7</td>
//!             <td>F (0.11)</td>
//!             <td>F (0.21)</td>
//!             <td>F (0.18</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//!
//! # Code Example:
//!
//! ```
//! use rank_biased_centroids::rbc;
//!
//! let r1 = vec!['A', 'D', 'B', 'C', 'G', 'F'];
//! let r2 = vec!['B', 'D', 'E', 'C'];
//! let r3 = vec!['A', 'B', 'D', 'C', 'G', 'F', 'E'];
//! let r4 = vec!['G', 'D', 'E', 'A', 'F', 'C'];
//! let p = 0.9;
//! let res = rbc(vec![r1, r2, r3, r4], p).unwrap();
//! let exp = vec![
//!     ('D', 0.35),
//!     ('C', 0.28),
//!     ('A', 0.27),
//!     ('B', 0.27),
//!     ('G', 0.23),
//!     ('E', 0.22),
//!     ('F', 0.18),
//! ];
//! for ((c, s), (ec, es)) in res.into_ranked_list_with_scores().into_iter().zip(exp.into_iter()) {
//!     assert_eq!(c, ec);
//!     approx::assert_abs_diff_eq!(s, es, epsilon = 0.005);
//! }
//! ```
//!
//! Weighted runs:
//!
//! ```rust
//! use rank_biased_centroids::rbc_with_weights;
//! let r1 = vec!['A', 'D', 'B', 'C', 'G', 'F'];
//! let r2 = vec!['B', 'D', 'E', 'C'];
//! let r3 = vec!['A', 'B', 'D', 'C', 'G', 'F', 'E'];
//! let r4 = vec!['G', 'D', 'E', 'A', 'F', 'C'];
//! let p = 0.9;
//! let run_weights = vec![0.3, 1.3, 0.4, 1.4];
//! let res = rbc_with_weights(vec![r1, r2, r3, r4],run_weights, p).unwrap();
//! let exp = vec![
//!     ('D', 0.30),
//!     ('E', 0.24),
//!     ('C', 0.23),
//!     ('B', 0.19),
//!     ('G', 0.19),
//!     ('A', 0.17),
//!     ('F', 0.13),
//! ];
//! for ((c, s), (ec, es)) in res.into_ranked_list_with_scores().into_iter().zip(exp.into_iter()) {
//!     assert_eq!(c, ec);
//!     approx::assert_abs_diff_eq!(s, es, epsilon = 0.005);
//! }
//! ```
//!
mod state;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RbcError {
    #[error("Persistance parameter p must be 0.0 <= p < 1.0")]
    InvalidPersistance,
    #[error("There need to be as many run weights as runs + not inf or nan")]
    InvalidRunWeights,
}

use state::RbcState;
use std::fmt::Debug;
use std::hash::Hash;

pub use state::RbcRankedList;

///
/// Main RBC function implementing the computation of Rank-Biased Centroids.
///
/// Returns the fused ranked list without scores.
///
/// # Example:
///
/// ```
/// use rank_biased_centroids::rbc;
///
/// let r1 = vec!['A', 'D', 'B', 'C', 'G', 'F'];
/// let r2 = vec!['B', 'D', 'E', 'C'];
/// let r3 = vec!['A', 'B', 'D', 'C', 'G', 'F', 'E'];
/// let r4 = vec!['G', 'D', 'E', 'A', 'F', 'C'];
/// let p = 0.9;
/// let res = rbc(vec![r1, r2, r3, r4], p).unwrap();
/// let exp = vec!['D','C','A','B','G','E','F'];
/// assert!(res.into_ranked_list().into_iter().eq(exp.into_iter()));
/// ```
/// # Errors
///
/// - Will return `Err` if `p` is not 0 <= p < 1
///
pub fn rbc<I>(
    input_rankings: I,
    p: f64,
) -> Result<RbcRankedList<<<I as IntoIterator>::Item as IntoIterator>::Item>, RbcError>
where
    I: IntoIterator,
    I::Item: IntoIterator,
    <<I as IntoIterator>::Item as IntoIterator>::Item: Eq + Hash + Debug,
{
    let mut rbc_state = RbcState::with_persistence(p)?;

    // iterate over all lists
    let ranked_list_iter = input_rankings.into_iter();
    for ranked_list in ranked_list_iter {
        for (rank, item) in ranked_list.into_iter().enumerate() {
            rbc_state.update(rank, item, None);
        }
    }

    // finalize
    Ok(rbc_state.into_result())
}

///
/// Main RBC function implementing the computation of Rank-Biased Centroids.
///
/// Returns the fused ranked list without scores.
///
/// # Example:
///
/// ```
/// use rank_biased_centroids::rbc_with_weights;
///
/// let r1 = vec!['A', 'D', 'B', 'C', 'G', 'F'];
/// let r2 = vec!['B', 'D', 'E', 'C'];
/// let r3 = vec!['A', 'B', 'D', 'C', 'G', 'F', 'E'];
/// let r4 = vec!['G', 'D', 'E', 'A', 'F', 'C'];
/// let p = 0.9;
/// let res = rbc_with_weights(vec![r1, r2, r3, r4],vec![0.3,1.3,0.4,1.4], p).unwrap();
/// let exp = vec!['D','E','C','B','G','A','F'];
/// let result = res.into_ranked_list();
/// assert!(result.into_iter().eq(exp.into_iter()));
/// ```
/// # Errors
///
/// - Will return `Err` if `p` is not 0 <= p < 1
/// - Will return `Err` if run weights len != num runs
/// - Will return `Err` if run weights are inf or NaN
///
pub fn rbc_with_weights<I>(
    input_rankings: I,
    run_weights: impl IntoIterator<Item = f64>,
    p: f64,
) -> Result<RbcRankedList<<<I as IntoIterator>::Item as IntoIterator>::Item>, RbcError>
where
    I: IntoIterator,
    I::Item: IntoIterator,
    <<I as IntoIterator>::Item as IntoIterator>::Item: Eq + Hash + Debug,
{
    let mut rbc_state = RbcState::with_persistence(p)?;

    // iterate over all lists
    let mut run_weights_iter = run_weights.into_iter();
    let mut ranked_list_iter = input_rankings.into_iter();
    for ranked_list in ranked_list_iter.by_ref() {
        let run_weight = match run_weights_iter.next() {
            None => return Err(RbcError::InvalidRunWeights),
            Some(w) if w.is_infinite() => return Err(RbcError::InvalidRunWeights),
            Some(w) if w.is_nan() => return Err(RbcError::InvalidRunWeights),
            Some(w) => Some(w),
        };
        for (rank, item) in ranked_list.into_iter().enumerate() {
            rbc_state.update(rank, item, run_weight);
        }
    }

    // more ranked list than weights!
    if ranked_list_iter.next().is_some() {
        return Err(RbcError::InvalidRunWeights);
    }

    // finalize
    Ok(rbc_state.into_result())
}
