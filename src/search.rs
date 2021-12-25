use std::clone::Clone;
use std::cmp::{Eq, Ord, Ordering};
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;

pub trait Graph {
    type Node: Display + Clone + Hash + Eq;
    type Edge: fmt::Display + Clone;
    fn null_edge() -> Self::Edge;
    fn start(&self) -> Self::Node;
    fn neighbors(&self, n: &Self::Node) -> Vec<(Self::Edge, usize, Self::Node)>;
    fn distance_to_goal(&self, n: &Self::Node) -> usize;
}

#[derive(Clone)]
/// Wrapper for holding objects in a priority queue, ordered by S
struct QueueEntry<S: PartialOrd, T>(S, T);

impl<S: PartialOrd, T> Eq for QueueEntry<S, T> {}
impl<S: PartialOrd, T> PartialEq for QueueEntry<S, T> {
    fn eq(&self, other: &QueueEntry<S, T>) -> bool {
        self.0.eq(&other.0)
    }
}
impl<S: PartialOrd, T> PartialOrd for QueueEntry<S, T> {
    fn partial_cmp(&self, other: &QueueEntry<S, T>) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}
impl<S: PartialOrd, T> Ord for QueueEntry<S, T> {
    fn cmp(&self, other: &QueueEntry<S, T>) -> Ordering {
        match other.0.partial_cmp(&self.0) {
            Some(r) => r,
            _ => panic!(),
        }
    }
}

pub fn dfs_search<G: Graph>(graph: &G) -> Option<Vec<(G::Edge, G::Node)>> {
    let mut visited = HashSet::new();
    let mut visits = 0;
    fn dfs<G: Graph>(
        visited: &mut HashSet<G::Node>,
        visits: &mut usize,
        graph: &G,
        current: G::Node,
    ) -> Option<Vec<(G::Edge, G::Node)>> {
        eprintln!("{}", &current);
        if graph.distance_to_goal(&current) == 0 {
            return Some(vec![]);
        }
        visited.insert(current.clone());
        if visited.len().count_ones() == 1 {
            eprintln!("Visited {}...", visited.len());
        }
        *visits += 1;
        for (edge, _, neighbor) in graph.neighbors(&current) {
            if !visited.contains(&neighbor) {
                if let Some(mut path) = dfs(visited, visits, graph, neighbor.clone()) {
                    path.push((edge, neighbor));
                    return Some(path);
                }
            }
        }
        visited.remove(&current);
        None
    };
    let result = dfs(&mut visited, &mut visits, graph, graph.start());
    println!("States visited: {}", visits);
    if let Some(mut path) = result {
        path.reverse();
        Some(path)
    } else {
        None
    }
}

pub fn a_star_search<G: Graph>(graph: &G) -> Option<(usize, Vec<(usize, G::Edge, G::Node)>)> {
    struct State<G: Graph> {
        visited: bool,
        prior: Option<G::Node>,
        prior_cost: usize,
        cost_guess: usize,
        dir: G::Edge,
    };
    let mut table = HashMap::new();
    let start = graph.start();
    let start_cost_guess = graph.distance_to_goal(&start);
    table.insert(
        start.clone(),
        State::<G> {
            visited: false,
            prior: None,
            prior_cost: 0,
            cost_guess: start_cost_guess,
            dir: G::null_edge(),
        },
    );
    let mut frontier = BinaryHeap::new();
    frontier.push(QueueEntry(start_cost_guess, start));
    let mut step: usize = 0;
    let mut min_cost = None;
    while let Some(QueueEntry(best_cost, ref current)) = frontier.pop() {
        //eprintln!("{}", current);
        step += 1;
        if let Some(old_cost) = min_cost {
            if old_cost > best_cost {
                min_cost = Some(best_cost);
            }
        } else {
            min_cost = Some(best_cost);
        }
        if step.count_ones() == 1 {
            eprintln!(
                "{} steps, cost {}, min cost {}...",
                step,
                best_cost,
                min_cost.unwrap()
            );
        }
        if best_cost == 0 || (step > 10 && frontier.is_empty()) {
            let mut path = vec![];
            let mut node = current;
            let cost = table.get(current).unwrap().prior_cost;
            let mut last_cost = 0;
            loop {
                let entry = table.get(node).unwrap();
                let cost = entry.prior_cost;
                path.push((last_cost - cost, entry.dir.clone(), node.clone()));
                if let &Some(ref next) = &entry.prior {
                    last_cost = cost;
                    node = next;
                } else {
                    break;
                }
            }
            path.reverse();
            println!("States visited: {}", table.len());
            return Some((cost, path));
        }
        let prior_cost = {
            let entry = table.get_mut(current).unwrap();
            if entry.visited {
                continue;
            }
            entry.visited = true;
            entry.prior_cost
        };
        for (dir, edge_cost, neighbor) in graph.neighbors(current) {
            let new_prior_cost = prior_cost + edge_cost;
            let cost_guess = new_prior_cost + graph.distance_to_goal(&neighbor);
            let candidate_entry = State::<G> {
                visited: false,
                prior: Some(current.clone()),
                prior_cost: new_prior_cost,
                cost_guess: cost_guess,
                dir: dir,
            };
            // if unseen or cost_guess is better, update/insert and requeue
            let should_enqueue = match table.entry(neighbor.clone()) {
                Occupied(occ) => {
                    let v = occ.into_mut();
                    if v.cost_guess > cost_guess {
                        *v = candidate_entry;
                        true
                    } else {
                        false
                    }
                }
                Vacant(vac) => {
                    vac.insert(candidate_entry);
                    true
                }
            };
            if should_enqueue {
                frontier.push(QueueEntry(cost_guess, neighbor));
            }
        }
    }
    println!("States visited: {}", table.len());
    None
}
