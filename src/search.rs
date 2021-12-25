use std::clone::Clone;
use std::cmp::{Eq, Ord, Ordering};
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

pub trait Graph {
    type Node: Clone + Hash + Eq;
    type Edge: fmt::Display + Clone;
    fn null_edge() -> Self::Edge;
    fn start(&self) -> Self::Node;
    fn goal(&self) -> Self::Node;
    fn neighbors(&self, n: &Self::Node) -> Vec<(Self::Edge, Self::Node)>;
    fn distance(&self, n1: &Self::Node, n2: &Self::Node) -> usize;
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
    let goal = graph.goal();
    let mut visits = 0;
    fn dfs<G: Graph>(
        visited: &mut HashSet<G::Node>,
        visits: &mut usize,
        graph: &G,
        current: G::Node,
        goal: &G::Node,
    ) -> Option<Vec<(G::Edge, G::Node)>> {
        if current == *goal {
            return Some(vec![]);
        }
        visited.insert(current.clone());
        *visits += 1;
        for (edge, neighbor) in graph.neighbors(&current) {
            if !visited.contains(&neighbor) {
                if let Some(mut path) = dfs(visited, visits, graph, neighbor.clone(), goal) {
                    path.push((edge, neighbor));
                    return Some(path);
                }
            }
        }
        visited.remove(&current);
        None
    };
    let result = dfs(&mut visited, &mut visits, graph, graph.start(), &goal);
    println!("States visited: {}", visits);
    if let Some(mut path) = result {
        path.reverse();
        Some(path)
    } else {
        None
    }
}

pub fn a_star_search<G: Graph>(graph: &G) -> Option<(usize, Vec<(G::Edge, G::Node)>)> {
    struct State<G: Graph> {
        visited: bool,
        prior: Option<G::Node>,
        prior_cost: usize,
        cost_guess: usize,
        dir: G::Edge,
    };
    let mut table = HashMap::new();
    let start = graph.start();
    let goal = graph.goal();
    let start_cost_guess = graph.distance(&start, &goal);
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
    while let Some(QueueEntry(_, ref current)) = frontier.pop() {
        if current == &goal {
            let mut path = vec![];
            let mut node = current;
            loop {
                let entry = table.get(node).unwrap();
                path.push((entry.dir.clone(), node.clone()));
                if let &Some(ref next) = &entry.prior {
                    node = next;
                } else {
                    break;
                }
            }
            path.reverse();
            println!("States visited: {}", table.len());
            let cost = table.get(current).unwrap().prior_cost;
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
        for (dir, neighbor) in graph.neighbors(current) {
            let new_prior_cost = prior_cost + graph.distance(current, &neighbor);
            let cost_guess = new_prior_cost + graph.distance(&neighbor, &goal);
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
