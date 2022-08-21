# Rank-Biased Centroids (RBC) [![Crates.io][crates-badge]][crates-url] [![Docs.rs][docs-badge]][docs-rs] [![MIT licensed][mit-badge]][mit-url]

[crates-badge]: https://img.shields.io/crates/v/rank_biased_centroids.svg
[crates-url]: https://crates.io/crates/rank_biased_centroids
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://opensource.org/licenses/MIT
[docs-rs]: https://docs.rs/rank_biased_centroids
[docs-badge]: https://img.shields.io/docsrs/rank_biased_centroids/0.2.2

The Rank-Biased Centroids (RBC) rank fusion method to combine multiple-rankings of objects.

This code implements the RBC rank fusion method, as described in:

```bibtex
@inproceedings{DBLP:conf/sigir/BaileyMST17,
  author    = {Peter Bailey and
               Alistair Moffat and
               Falk Scholer and
               Paul Thomas},
  title     = {Retrieval Consistency in the 
               Presence of Query Variations},
  booktitle = {Proc of {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval,
  pages     = {395--404},
  publisher = {{ACM}},
  year      = {2017},
  url       = {https://doi.org/10.1145/3077136.3080839},
  doi       = {10.1145/3077136.3080839},
  timestamp = {Wed, 25 Sep 2019 16:43:14 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/BaileyMST17.bib},
}
```

The fundamental step in the working of RBC is the usage a `persistence` parameter (`p` or `phi`) to to fusion multiple ranked lists based only on rank information. Larger values of `p` give higher importance to elements at the top of each ranking. From the paper:

> As extreme values, consider `p = 0` and `p = 1`. When `p = 0`, the agents only ever examine the first item in each of the input rankings, and the fused output is by decreasing score of firstrst preference; this is somewhat akin to a first-past-the-post election regime. When `p = 1`, each agent examines the whole of every list, and the fused ordering is determined by the number of lists that contain each item – a kind of "popularity count" of each item across the input sets. In between these extremes, the expected depth reached by the agents viewing the rankings is given by `1/(1 − p)`. For example, when `p = 0.9`, on average the first 10 items in each ranking are being used to contribute to the fused ordering; of course, in aggregate, across the whole universe of agents, all of the items in every ranking contribute to the overall outcome.

More from the paper:

> Each item at rank 1 <= x <= n when the rankings are over n items, we suggest that a geometrically decaying weight function be employed, with the distribution of d over depths x given by (1 − p) p^{x-1} for some value 0 <= p <= 1 determined by considering the purpose for which the fused ranking is being constructed. 

# Example fusion

For example (also taken from the paper) for diffent rank orderings (R1-R4) of items A-G:

|Rank| R1  | R2  | R3  | R4  |
| ---| --- | --- | --- | --- |
| 1  |  A  |  B  |  A  |  G  |
| 2  |  D  |  D  |  B  |  D  |
| 3  |  B  |  E  |  D  |  E  |
| 4  |  C  |  C  |  C  |  A  |
| 5  |  G  |  -  |  G  |  F  |
| 6  |  F  |  -  |  F  |  C  |
| 7  |  -  |  -  |  E  |  -  |

Depending on the persistence parameter `p` will result in different output orderings based on each items accumulated weights:

|Rank|   p=0.6   | p=0.8   | p=0.9   |
| ---| ------    | ------  | ------  |
| 1  |  A(0.89)  | D(0.61) | D(0.35) |
| 2  |  D(0.86)  | A(0.50) | C(0.28) |
| 3  |  B(0.78)  | B(0.49) | A(0.27) |
| 4  |  G(0.50)  | C(0.37) | B(0.27) |
| 5  |  E(0.31)  | G(0.37) | G(0.23) |
| 6  |  C(0.29)  | E(0.31) | E(0.22) |
| 7  |  F(0.11)  | F(0.21) | F(0.18) |

# Code Example:



```rust
use rank_biased_centroids::rbc_with_scores;
let r1 = vec!['A', 'D', 'B', 'C', 'G', 'F'];
let r2 = vec!['B', 'D', 'E', 'C'];
let r3 = vec!['A', 'B', 'D', 'C', 'G', 'F', 'E'];
let r4 = vec!['G', 'D', 'E', 'A', 'F', 'C'];
let p = 0.9;
let res = rbc_with_scores(vec![r1, r2, r3, r4], p).unwrap();
let exp = vec![
    ('D', 0.35),
    ('C', 0.28),
    ('A', 0.27),
    ('B', 0.27),
    ('G', 0.23),
    ('E', 0.22),
    ('F', 0.18),
];
for ((c, s), (ec, es)) in res.into_iter().zip(exp.into_iter()) {
    assert_eq!(c, ec);
    approx::assert_abs_diff_eq!(s, es, epsilon = 0.005);
}
```

# License

MIT
