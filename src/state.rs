use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

const VALID_P_RANGE: std::ops::Range<f64> = 0.0..1.0;

pub(crate) struct RbcState<Item: Eq + Hash + Debug> {
    // precomputed weights for different ranks
    weights: Vec<f64>,
    // accumulated item weights
    item_weights: HashMap<Item, f64>,
    // persitstence
    persistence: f64,
}

impl<Item: Eq + Hash + Debug> RbcState<Item> {
    // Initialize the RBO state with persistance `p`
    pub(crate) fn with_persistence(persistence: f64) -> Result<Self, crate::RbcError> {
        if !VALID_P_RANGE.contains(&persistence) {
            return Err(crate::RbcError::InvalidPersistance);
        }
        let mut w = 1.0 - persistence;
        let weights = (0..10000i32)
            .map(|_| {
                let pw = w;
                w *= persistence;
                pw
            })
            .collect();
        Ok(Self {
            persistence,
            weights,
            item_weights: HashMap::new(),
        })
    }

    // Update the RBO state with two new elements.
    pub(crate) fn update(&mut self, rank: usize, item: Item, run_weight: Option<f64>) {
        while self.weights.len() <= rank {
            let last_weight = self.weights.last().expect("can't fail");
            let new_last = last_weight * self.persistence;
            self.weights.push(new_last);
        }
        let w = *self.weights.get(rank).expect("this can't fail now");
        let weight = if let Some(rw) = run_weight { w * rw } else { w };
        self.item_weights
            .entry(item)
            .and_modify(|e| *e += weight)
            .or_insert(weight);
    }

    // we extrapolate the final RBO value and compute the residual
    pub(crate) fn into_result(self) -> RbcRankedList<Item> {
        let mut items: Vec<(Item, f64)> = self.item_weights.into_iter().collect();
        items.sort_by(|a, b| b.1.total_cmp(&a.1));
        RbcRankedList { items }
    }
}

/// contains the rank fused list of items in order
#[derive(Debug)]
pub struct RbcRankedList<Item: Eq + Hash + Debug> {
    items: Vec<(Item, f64)>,
}

impl<Item: Eq + Hash + Debug> RbcRankedList<Item> {
    /// return the fused ranked list of items without scores
    #[must_use]
    pub fn into_ranked_list(self) -> Vec<Item> {
        self.items.into_iter().map(|(item, _)| item).collect()
    }

    /// return the fused ranked list of items with scores
    #[must_use]
    pub fn into_ranked_list_with_scores(self) -> Vec<(Item, f64)> {
        self.items
    }
}
