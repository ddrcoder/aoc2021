#[macro_use]
extern crate scan_fmt;
extern crate clap;
#[macro_use]
extern crate vectrix;
#[macro_use]
extern crate ndarray;

use clap::Parser;
use ndarray::{Array2, Array3};
use std::cmp::{max, min, Ordering};
use std::hash::Hash;
use std::time::Instant;
//use std::hash::Hash;
use std::{collections::VecDeque, fmt::Debug};

use ndarray::Array;
use regex::{Captures, Regex};
use std::collections::{hash_map::HashMap, hash_set::HashSet};
use std::fs::File;
use std::io::Read;
use std::iter::once;
//use std::iter::{once, Iterator};
use vectrix::{Matrix, Vector};

fn day1(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    // Lines -> numbers.
    let nums: Vec<usize> = lines.iter().map(|s| s.parse().ok().unwrap_or(0)).collect();
    // Numbers -> windows of width 3 or 1.
    let windows = nums.windows(if gold { 3 } else { 1 });
    // Sums of each window.
    let sums: Vec<usize> = windows.map(|window| window.iter().cloned().sum()).collect();
    // Adjacent pairs of sums.
    let pairs = sums.windows(2);
    let increasing_pairs = pairs.filter(|w| w[1] > w[0]);
    increasing_pairs.count()
}

fn day2(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let instructions = lines
        .into_iter()
        .flat_map(|s| scan_fmt!(s, "{} {}", String, usize).ok().into_iter());
    let (mut x, mut y, mut a) = (0, 0, 0);
    for (d, n) in instructions {
        match &d[..] {
            "forward" => {
                x += n;
                if gold {
                    y += n * a;
                }
            }
            "down" => {
                a += n;
            }
            "up" => {
                a -= n;
            }
            _ => {}
        }
    }
    if !gold {
        y = a;
    }
    x * y
}

fn day3(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let w = lines[0].len();
    let start: Vec<usize> = lines
        .iter()
        .map(|s| usize::from_str_radix(s, 2).ok().unwrap())
        .collect();
    let ones_ge_zeros = |nums: &[usize], b| {
        let c = nums.iter().cloned().filter(|x| (x >> b) & 1 == 1).count();
        c * 2 >= nums.len()
    };
    if gold {
        let (mut o2, mut co2) = (0, 0);
        for (output, want_most) in [(&mut o2, false), (&mut co2, true)] {
            let mut remaining: Vec<_> = start.clone();
            for b in (0..w).rev() {
                if remaining.len() <= 1 {
                    break;
                }
                let keep_ones = ones_ge_zeros(&remaining, b) == want_most;
                remaining = remaining
                    .into_iter()
                    .filter(|x| ((x >> b) & 1 == 1) == keep_ones)
                    .collect();
            }
            *output = remaining[0];
        }
        o2 * co2
    } else {
        let gamma = (0..w)
            .rev()
            .map(|b| ones_ge_zeros(&start, b))
            .fold(0, |v, is_one| v * 2 + (is_one as usize));
        let mask = (1 << w) - 1;
        let epsilon = !gamma & mask;
        gamma * epsilon
    }
}

fn check_board(board: &mut [[i64; 5]; 5], n: i64) -> Option<i64> {
    let mut sum = 0;
    for row in board.iter_mut() {
        for num in row.iter_mut() {
            if *num == n {
                *num *= -1;
            } else if *num > 0 {
                sum += *num;
            }
        }
    }
    for row in 0..5 {
        for col in 0..5 {
            if board[row][col] > 0 {
                break;
            }
            if col == 4 {
                return Some(sum);
            }
        }
    }
    for col in 0..5 {
        for row in 0..5 {
            if board[row][col] > 0 {
                break;
            }
            if row == 4 {
                return Some(sum);
            }
        }
    }

    None
}

fn collect_array<T: Default + Copy, Iter: Iterator<Item = T>, const N: usize>(
    mut iter: Iter,
) -> Option<[T; N]> {
    let mut a = [Default::default(); N];
    for i in 0..N {
        if let Some(value) = iter.next() {
            a[i] = value;
        }
    }
    if let None = iter.next() {
        Some(a)
    } else {
        None
    }
}

fn day4(_lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    let numbers: Vec<i64> = groups[0][0]
        .split(',')
        .flat_map(|n| n.parse().ok())
        .collect();
    let mut boards: Vec<[[i64; 5]; 5]> =
        groups[1..]
            .iter()
            .map(|lines| {
                collect_array(lines.iter().map(|line| {
                    collect_array(line.split(' ').flat_map(|n| n.parse().ok())).unwrap()
                }))
                .unwrap()
            })
            .collect();
    let mut last = 0;
    let mut won = HashSet::new();
    for n in numbers {
        for (i, board) in boards.iter_mut().enumerate() {
            if let Some(sum) = check_board(board, n) {
                if won.insert(i) {
                    let score = (sum * n) as usize;
                    if !gold {
                        return score;
                    }
                    last = score;
                }
            }
        }
    }
    return last;
}

fn day5(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let lines = lines.iter().map(|line| {
        scan_fmt!(line, "{},{} -> {},{}", i64, i64, i64, i64)
            .ok()
            .unwrap()
    });
    let mut grid: HashMap<(i64, i64), usize> = HashMap::new();
    for (x1, y1, x2, y2) in lines {
        let points: Vec<(i64, i64)> = if x1 == x2 {
            let (y1, y2) = if y1 > y2 { (y2, y1) } else { (y1, y2) };
            (y1..=y2).map(|y| (x1, y)).collect()
        } else if y1 == y2 {
            let (x1, x2) = if x1 > x2 { (x2, x1) } else { (x1, x2) };
            (x1..=x2).map(|x| (x, y1)).collect()
        } else if gold {
            if (x1 - x2).abs() >= (y1 - y2).abs() {
                let (x1, y1, x2, y2) = if x1 > x2 {
                    (x2, y2, x1, y1)
                } else {
                    (x1, y1, x2, y2)
                };
                (x1..=x2)
                    .map(|x| (x, (x - x1) * (y2 - y1) / (x2 - x1) + y1))
                    .collect()
            } else {
                let (x1, x2, y1, y2) = if y1 > y2 {
                    (x2, y2, x1, y1)
                } else {
                    (x1, x2, y1, y2)
                };
                (y1..=y2)
                    .map(|y| ((y - y1) * (x2 - x1) / (y2 - y1) + x1, y))
                    .collect()
            }
        } else {
            continue;
        };
        for (x, y) in points {
            hist_inc(&mut grid, (x, y), 1);
        }
    }
    grid.values().filter(|v| **v > 1).count()
}

fn day6(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let mut birthday_counts = [0; 265];
    for d in lines[0].split(',') {
        let day: usize = d.parse().ok().unwrap();
        birthday_counts[day] += 1;
    }
    let end = if gold { 256 } else { 80 };
    for day in 0..end {
        let n = birthday_counts[day];
        birthday_counts[day + 7] += n;
        birthday_counts[day + 9] += n;
    }
    birthday_counts[end..].into_iter().sum()
}

fn day7(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let mut crabs: Vec<usize> = lines[0].split(',').flat_map(|n| n.parse().ok()).collect();
    crabs.sort();
    let mut crabs = &crabs[..];
    #[derive(Debug)]
    struct End {
        c: usize,
        dc: usize,
        n: usize,
    }
    let mut l = End { c: 0, dc: 0, n: 0 };
    let mut r = End { c: 0, dc: 0, n: 0 };
    while crabs.len() > 2 {
        let dl = crabs[1] - crabs[0];
        let dr = crabs[crabs.len() - 1] - crabs[crabs.len() - 2];
        let (next_crabs, d, end) = if r.dc < l.dc {
            (&crabs[0..(crabs.len() - 1)], dr, &mut r)
        } else {
            (&crabs[1..], dl, &mut l)
        };
        crabs = next_crabs;
        end.n += 1;
        if gold {
            // TODO: Calculus.
            for s in 0..d {
                end.dc += end.n;
                end.c += end.dc;
            }
        } else {
            end.dc += 1;
            end.c += end.dc * d;
        }
    }
    // Setup derivatives to be ready to include crab at either end.
    l.n += 1;
    l.dc += l.n;
    r.n += 1;
    r.dc += r.n;
    let (mut pl, mut pr) = (crabs[0], crabs[1]);
    while pl < pr {
        if r.dc < l.dc {
            pr -= 1;
            r.c += r.dc;
            if gold {
                r.dc += r.n;
            }
        } else {
            pl += 1;
            l.c += l.dc;
            if gold {
                l.dc += l.n;
            }
        }
    }

    l.c + r.c
}

fn day8(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = lines
        .iter()
        .map(|line| {
            line.split(" | ").map(|set| {
                set.split(' ')
                    .map(|s| {
                        s.chars()
                            .map(|ch| 1 << ((ch as u8 - 'a' as u8) as usize))
                            .fold(0, |n, b| (n | b))
                    })
                    .collect()
            })
        })
        .map(|mut parts| (parts.next().unwrap(), parts.next().unwrap()))
        .collect();
    pairs
        .into_iter()
        .map(|(samples, output)| {
            fn find<F: FnMut(u8) -> bool>(samples: &[u8], mut test: F) -> u8 {
                let mut matches = samples.iter().cloned().filter(|v| test(*v));
                let first = matches.next().unwrap();
                if let Some(second) = matches.next() {
                    panic!("matched {} and {}", first, second);
                }
                first
            }
            let samples = &samples[..];
            let mut codes = [0; 10];
            codes[1] = find(samples, |v| v.count_ones() == 2);
            codes[4] = find(samples, |v| v.count_ones() == 4);
            codes[7] = find(samples, |v| v.count_ones() == 3);
            codes[8] = find(samples, |v| v.count_ones() == 7);
            codes[3] = find(samples, |v| {
                v.count_ones() == 5 && (v ^ codes[7]).count_ones() == 2
            });
            codes[9] = find(samples, |v| {
                v.count_ones() == 6 && (v ^ codes[3]).count_ones() == 1
            });
            codes[5] = find(samples, |v| {
                v.count_ones() == 5 && (v | codes[7]) == codes[9]
            });
            codes[6] = find(samples, |v| {
                v.count_ones() == 6 && (v | codes[1]) == codes[8]
            });
            codes[0] = find(samples, |v| {
                v.count_ones() == 6 && (v & (codes[9] ^ codes[6])).count_ones() == 2
            });
            codes[2] = find(samples, |v| {
                v.count_ones() == 5 && (v | codes[4]) == codes[8]
            });
            output
                .iter()
                .map(|code| {
                    codes
                        .iter()
                        .enumerate()
                        .find_map(|(i, item)| if code == item { Some(i) } else { None })
                        .unwrap()
                })
                .fold(0, |n, d| {
                    if gold {
                        n * 10 + d
                    } else {
                        n + match d {
                            1 | 4 | 7 | 8 => 1,
                            _ => 0,
                        }
                    }
                })
        })
        .sum()
}

fn day9(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    fn fill(land: &mut Vec<Vec<u8>>, x: usize, y: usize) -> usize {
        let here = &mut land[y][x];
        let mut s = 0;
        if *here < 9 {
            s += 1;
            *here = 10;
            if x > 0 {
                s += fill(land, x - 1, y)
            }
            if x + 1 < land[y].len() {
                s += fill(land, x + 1, y);
            }

            if y > 0 {
                s += fill(land, x, y - 1)
            }

            if y + 1 < land.len() {
                s += fill(land, x, y + 1);
            }
        }
        s
    }

    let mut land: Vec<Vec<_>> = lines
        .iter()
        .map(|line| line.chars().map(|ch| (ch as u8 - '0' as u8)).collect())
        .collect();
    let mut starts = vec![];
    for y in 0..land.len() {
        let row = &land[y];
        for x in 0..row.len() {
            let here = row[x];
            if (x > 0 && row[x - 1] <= here)
                || (x + 1 < row.len() && row[x + 1] <= here)
                || (y > 0 && land[y - 1][x] <= here)
                || (y + 1 < land.len() && land[y + 1][x] <= here)
            {
                continue;
            }
            starts.push((x, y));
        }
    }
    if gold {
        let mut areas = vec![0; starts.len()];
        for (i, (x, y)) in starts.into_iter().enumerate() {
            areas[i] = fill(&mut land, x, y);
        }
        areas.sort();
        areas[(areas.len() - 3)..].into_iter().product()
    } else {
        starts
            .into_iter()
            .map(|(x, y)| land[y][x] as usize + 1)
            .sum()
    }
}

fn day10(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let nested = "([{<>}])";
    let e_scores = [3, 57, 1197, 25137];
    let mut scores: Vec<_> = lines
        .iter()
        .filter_map(|line| {
            let mut stack = vec![];
            for ch in line.chars() {
                if let Some(found) = nested.find(ch) {
                    let other = nested.len() - 1 - found;
                    if found < 4 {
                        stack.push(found);
                    } else if stack.pop() != Some(other) {
                        return if gold { None } else { Some(e_scores[other]) };
                    }
                } else {
                    break; // non paren
                }
            }
            if gold {
                Some(stack.into_iter().rev().fold(0, |s, d| s * 5 + d + 1))
            } else {
                None
            }
        })
        .collect();
    if gold {
        scores.sort();
        *scores.get(&scores.len() / 2).unwrap_or(&0)
    } else {
        scores.into_iter().sum()
    }
}

fn day11(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let mut grid: Vec<Vec<_>> = lines
        .iter()
        .map(|line| line.chars().map(|ch| (ch as u8 - '0' as u8)).collect())
        .collect();
    let mut flashes = 0;
    for i in 0.. {
        if false {
            eprintln!("{}", i);
            for y in 0..grid.len() {
                if grid.len() > 40 {
                    break;
                }
                for x in 0..grid[y].len() {
                    eprint!("{}", (grid[y][x] + '0' as u8) as char);
                }
                eprintln!();
            }
            eprintln!();
        }
        for y in 0..grid.len() {
            for x in 0..grid[y].len() {
                grid[y][x] += 1;
            }
        }
        let mut change = true;

        while change {
            change = false;
            for y in 0..grid.len() {
                for x in 0..grid[y].len() {
                    if grid[y][x] == 10 {
                        grid[y][x] = 11;
                        change = true;
                        for ny in (if y >= 1 { y - 1 } else { y })..=(if y + 1 < grid.len() {
                            y + 1
                        } else {
                            y
                        }) {
                            for nx in (if x >= 1 { x - 1 } else { x })..=(if x + 1 < grid[y].len() {
                                x + 1
                            } else {
                                x
                            }) {
                                let p = &mut grid[ny][nx];
                                if *p < 10 {
                                    *p += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        for y in 0..grid.len() {
            for x in 0..grid[y].len() {
                let p = &mut grid[y][x];
                if *p > 9 {
                    flashes += 1;
                    *p = 0;
                }
            }
        }
        if gold {
            if grid.iter().all(|line| line.iter().all(|v| *v == 0)) {
                return i + 1;
            }
        } else if i == 100 {
            break;
        }
    }
    flashes
}

fn count_paths<'a>(
    current: &'a str,
    edges: &'a HashMap<&'a str, HashSet<&'a str>>,
    visited_small: &mut HashSet<&'a str>,
    spent_small: bool,
) -> usize {
    if current == "end" {
        1
    } else if let Some(nexts) = edges.get(current) {
        let mut n = 0;
        for next in nexts.iter() {
            let big = next.chars().all(|ch| ch.is_ascii_uppercase());
            let spend = if !big && !visited_small.insert(*next) {
                if !spent_small && next != &"start" {
                    true
                } else {
                    continue;
                }
            } else {
                false
            };
            n += count_paths(next, edges, visited_small, spent_small || spend);
            if !big && !spend {
                visited_small.remove(next);
            }
        }
        n
    } else {
        0
    }
}

fn day12(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let mut edges: HashMap<&str, HashSet<&str>> = HashMap::new();
    for line in lines {
        let mut side = line.split('-');
        let n0 = side.next().unwrap();
        let n1 = side.next().unwrap();
        edges.entry(n0).or_insert(HashSet::new()).insert(n1);
        edges.entry(n1).or_insert(HashSet::new()).insert(n0);
    }
    count_paths("start", &edges, &mut ["start"].into(), !gold)
}

fn day13(_lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    let mut points: HashSet<_> = groups[0]
        .iter()
        .map(|line| scan_fmt!(line, "{},{}", i64, i64).ok().unwrap())
        .collect();
    let folds: Vec<_> = groups[1]
        .iter()
        .map(|fold| scan_fmt!(fold, "fold along {}={}", char, i64).ok().unwrap())
        .collect();
    for (axis, cut) in folds {
        points = points
            .into_iter()
            .map(|(x, y)| match axis {
                'x' => (if x < cut { x } else { cut * 2 - x }, y),
                'y' => (x, if y < cut { y } else { cut * 2 - y }),
                _ => panic!(),
            })
            .collect();
        if !gold {
            return points.len();
        }
    }
    if false {
        for y in 0..=10 {
            println!();
            for x in 0..=40 {
                print!("{}", if points.contains(&(x, y)) { "##" } else { "  " });
            }
        }
    }
    0
}

fn hist_inc<K: Eq + Hash>(hist: &mut HashMap<K, usize>, k: K, n: usize) {
    *hist.entry(k).or_insert(0) += n;
}
fn hist_make<K: Eq + Hash, I: Iterator<Item = K>>(keys: I) -> HashMap<K, usize> {
    let mut hist = HashMap::new();
    keys.for_each(|k| hist_inc(&mut hist, k, 1));
    hist
}

fn day14(_lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    let rules: HashMap<_, _> = groups[1]
        .iter()
        .flat_map(|line| scan_fmt!(line, "{/./}{/./} -> {/./}", char, char, char).ok())
        .map(|(a, b, m)| ((a, b), m))
        .collect();
    let s = groups[0][0];
    let mut char_counts = hist_make(s.chars());
    let pairs = s.chars().zip(s.chars().skip(1));
    let mut pair_counts = hist_make(pairs);
    for _step in 0..(if gold { 40 } else { 10 }) {
        let mut pair_counts_new = HashMap::new();
        for ((a, b), n) in pair_counts {
            if let Some(&m) = rules.get(&(a, b)) {
                hist_inc(&mut char_counts, m, n);
                hist_inc(&mut pair_counts_new, (a, m), n);
                hist_inc(&mut pair_counts_new, (m, b), n);
            }
        }
        pair_counts = pair_counts_new;
    }
    char_counts.values().max().unwrap() - char_counts.values().min().unwrap()
}

fn bfs(grid: &mut Vec<Vec<(u8, Option<usize>)>>) -> Option<usize> {
    grid[0][0].1 = Some(0);
    let mut touched: HashSet<_> = HashSet::from([(0, 0)]);
    while !touched.is_empty() {
        fn merge(neighbor: &mut (u8, Option<usize>), here: usize) -> bool {
            if let (step, Some(ref mut cost)) = neighbor {
                if *cost > here + *step as usize {
                    *cost = here + *step as usize;
                    true
                } else {
                    false
                }
            } else {
                neighbor.1 = Some(here + neighbor.0 as usize);
                true
            }
        }
        let mut next = HashSet::new();
        for (x, y) in touched {
            let here = grid[y][x].1.unwrap();
            if x > 0 && merge(&mut grid[y][x - 1], here) {
                next.insert((x - 1, y));
            }
            if x + 1 < grid[y].len() && merge(&mut grid[y][x + 1], here) {
                next.insert((x + 1, y));
            };
            if y > 0 && merge(&mut grid[y - 1][x], here) {
                next.insert((x, y - 1));
            }
            if y + 1 < grid.len() && merge(&mut grid[y + 1][x], here) {
                next.insert((x, y + 1));
            };
        }
        touched = next;
    }
    let y = grid.len() - 1;
    grid[y][grid[y].len() - 1].1
}

fn day15(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let mut grid: Vec<Vec<(u8, Option<usize>)>> = lines
        .iter()
        .map(|line| {
            line.chars()
                .map(|ch| ((ch as u8 - '0' as u8), None))
                .collect()
        })
        .collect();
    if gold {
        grid = (0..5)
            .flat_map(|addy| {
                grid.clone().into_iter().map(move |row| {
                    (0..5)
                        .flat_map(|addx| {
                            row.clone()
                                .into_iter()
                                .map(move |(step, _)| ((step + addx + addy - 1) % 9 + 1, None))
                        })
                        .collect()
                })
            })
            .collect();
    }

    bfs(&mut grid).unwrap()
}

fn read(i: &mut usize, bits: &[u8], n: usize) -> usize {
    let mut r = 0;
    for _ in 0..n {
        r = r * 2 + bits[*i] as usize;
        *i += 1;
    }
    r
}
fn process(bits: &[u8], i: &mut usize, limit: Option<usize>) -> Vec<usize> {
    let mut n = 0;
    let mut ret = vec![];
    while *i < bits.len() && (limit.is_none() || n < limit.unwrap()) {
        n += 1;
        let version = read(i, bits, 3);
        //eprintln!("Read version {}", version);
        ret.push(match read(i, bits, 3) {
            4 => {
                let mut v = 0;
                while read(i, bits, 1) == 1 {
                    v = (v << 4) + read(i, bits, 4);
                    //eprintln!("Reading literal {}", v);
                }
                v = (v << 4) + read(i, bits, 4);
                //eprintln!("Read literal {}", v);
                //while *i & 3 != 0 { *i += 1; }
                v
            }
            a => {
                //eprintln!("Read op {}", a);
                //eprintln!("{{");

                let inner = if read(i, bits, 1) == 0 {
                    let length = read(i, bits, 15);
                    //eprintln!("Bit length: {}", length);
                    let end = *i + length;
                    let ret = process(&bits[0..end], i, None);
                    //assert_eq!(*i, end);
                    ret
                } else {
                    let n = read(i, bits, 11);
                    //eprintln!("Count: {}", n);
                    process(bits, i, Some(n))
                };
                //eprintln!("}}");
                let values = inner.iter().cloned();
                let (op, r) = match a {
                    0 => ("sum", values.sum()),
                    1 => ("", values.product()),
                    2 => ("", values.min().unwrap()),
                    3 => ("", values.max().unwrap()),
                    5 => ("gt", {
                        assert_eq!(inner.len(), 2);
                        if inner[0] > inner[1] {
                            1
                        } else {
                            0
                        }
                    }),
                    6 => ("lt", {
                        assert_eq!(inner.len(), 2);
                        if inner[0] < inner[1] {
                            1
                        } else {
                            0
                        }
                    }),
                    7 => ("eq", {
                        assert_eq!(inner.len(), 2);
                        if inner[0] == inner[1] {
                            1
                        } else {
                            0
                        }
                    }),
                    o => {
                        panic!("unexpected: {}", o)
                    }
                };
                //eprintln!("{} of {:?} is {}", op, &inner, r);
                r
            }
        });
    }
    ret
}

fn day16(lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    let bits: Vec<_> = lines[0]
        .chars()
        .map(|ch| {
            if ch >= 'A' {
                ch as u8 - 'A' as u8 + 10
            } else if ch >= '0' {
                ch as u8 - '0' as u8
            } else {
                panic!();
            }
        })
        .flat_map(|n| (0..4).rev().map(move |b| (n >> b) & 1))
        .collect();
    let r = process(&bits, &mut 0, Some(1));
    assert_eq!(r.len(), 1);
    //eprintln!("----------");
    r[0]
}

fn day17(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let (xl, xh, yl, yh) = scan_fmt!(
        lines[0],
        "target area: x={}..{}, y={}..{}",
        i64,
        i64,
        i64,
        i64
    )
    .ok()
    .unwrap();
    // TODO: Actually inform the plausible ranges from inputs; unsure if x range
    // is ever negative.
    let mut solutions = (-200..500)
        .rev()
        .flat_map(|vy0| (0..400).map(move |vx0| (vx0, vy0)))
        .flat_map(|(vx0, vy0): (i64, i64)| {
            let (mut x, mut y) = (0, 0);
            let (mut vx, mut vy) = (vx0, vy0);
            let mut max_y = if vy0 <= 0 { Some(0) } else { None };
            loop {
                x += vx;
                y += vy;
                vy -= 1;
                vx -= vx.signum();
                if vy == 0 {
                    max_y = Some(y);
                }
                if y < yl || x > xh || (vx == 0 && x < xl) {
                    return None;
                } else if x >= xl && y <= yh {
                    return Some((vx0, vy0, max_y.unwrap()));
                }
            }
        });
    if gold {
        solutions.count()
    } else {
        solutions.next().unwrap().2 as usize
    }
}

#[derive(Debug, Clone)]
enum Snail {
    Leaf(u8),
    Pair(Box<Snail>, Box<Snail>),
}

impl Snail {
    fn parse(mut text: &str) -> Snail {
        fn rec(text: &mut &str) -> Snail {
            if text.starts_with('[') {
                *text = &text[1..];
                let l = rec(text);
                assert!(text.starts_with(','));
                *text = &text[1..]; // ,
                let r = rec(text);
                assert!(text.starts_with(']'));
                *text = &text[1..]; // ']'
                Snail::Pair(Box::new(l), Box::new(r))
            } else {
                let n = text.chars().next().unwrap() as u8 - '0' as u8;
                *text = &text[1..];
                Snail::Leaf(n)
            }
        }

        rec(&mut text)
    }

    pub fn add(a: Snail, b: Snail) -> Snail {
        fn explode(snail: &mut Snail) -> bool {
            fn rec(snail: &mut Snail, depth: usize) -> (Option<u8>, Option<u8>, bool) {
                match snail {
                    Snail::Pair(a, b) => match (a.as_mut(), b.as_mut()) {
                        (Snail::Leaf(l), Snail::Leaf(r)) if depth >= 4 => {
                            let ret = (Some(*l), Some(*r), true);
                            *snail = Snail::Leaf(0);
                            ret
                        }
                        (left, right) => {
                            let (left_carry, right_carry, exploded) = rec(left, depth + 1);
                            if exploded {
                                if let Some(right_add) = right_carry {
                                    right.leftmost_add(right_add);
                                }
                                (left_carry, None, true)
                            } else {
                                let (left_carry, right_carry, exploded) = rec(right, depth + 1);
                                if exploded {
                                    if let Some(left_add) = left_carry {
                                        left.rightmost_add(left_add);
                                    }
                                    (None, right_carry, true)
                                } else {
                                    (None, None, false)
                                }
                            }
                        }
                    },
                    _ => (None, None, false),
                }
            }
            rec(snail, 0).2
        }

        fn split(snail: &mut Snail) -> bool {
            match snail {
                Snail::Pair(left, right) => split(&mut *left) || split(&mut *right),
                Snail::Leaf(n) if *n >= 10 => {
                    *snail = Snail::Pair(
                        Box::new(Snail::Leaf(*n / 2)),
                        Box::new(Snail::Leaf(*n - *n / 2)),
                    );
                    true
                }
                _ => false,
            }
        }

        let mut snail = Snail::Pair(Box::new(a), Box::new(b));
        loop {
            if explode(&mut snail) {
                //eprintln!("Exploded: {:?}", &snail);
            } else if split(&mut snail) {
                //eprintln!("Split: {:?}", &snail);
            } else {
                //eprintln!("Stable! {:?}", &snail);
                return snail;
            }
        }
    }

    fn leftmost_add(&mut self, add: u8) -> bool {
        match self {
            Snail::Pair(left, _) => left.leftmost_add(add),
            Snail::Leaf(n) => {
                *n += add;
                true
            }
        }
    }

    fn rightmost_add(&mut self, add: u8) {
        match self {
            Snail::Pair(_, right) => right.rightmost_add(add),
            Snail::Leaf(n) => {
                *n += add;
            }
        }
    }

    fn magnitude(&self) -> usize {
        match self {
            Snail::Pair(left, right) => left.magnitude() * 3 + right.magnitude() * 2,
            Snail::Leaf(n) => *n as usize,
        }
    }
}

fn self_cross<T: Copy, I: Iterator<Item = T> + Clone>(items: I) -> impl Iterator<Item = (T, T)> {
    items
        .clone()
        .flat_map(move |a| items.clone().map(move |b| (a, b)))
}

fn pairs<'a, T: 'a, I: Iterator<Item = &'a T> + 'a + Clone>(
    items: I,
) -> impl Iterator<Item = (&'a T, &'a T)> {
    self_cross(items).filter(|(a, b)| (*a as *const T) < (*b as *const T))
}

fn day18(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let snails: Vec<_> = lines.iter().cloned().map(Snail::parse).collect();
    if gold {
        pairs(snails.iter())
            .flat_map(|(a, b)| once((a, b)).chain(once((b, a))))
            .map(|(a, b)| Snail::add(a.clone(), b.clone()).magnitude())
            .max()
            .unwrap()
    } else {
        snails.into_iter().reduce(Snail::add).unwrap().magnitude()
    }
}

type Position = Vector<i16, 4>;
type Direction = Vector<i16, 4>;
type Transform = Matrix<i16, 4, 4>;
#[derive(Clone)]
struct Scanner {
    beacons: Vec<Position>,
    edges: HashMap<Direction, (Position, Position)>,
    n: usize,
}

impl Scanner {
    fn parse(lines: &[&str]) -> Self {
        let beacons: Vec<_> = lines
            .iter()
            .map(|line| scan_fmt!(line, "{},{},{}", i16, i16, i16).ok().unwrap())
            .map(|(a, b, c)| Position::from([a, b, c, 1]))
            .collect();
        let edges: HashMap<_, _> = pairs(beacons.iter())
            .map(|(a, b)| (Self::diff_invariant(a, b), (a.clone(), b.clone())))
            .collect();
        // Non-unique edges possible, but unsupported.
        assert_eq!(edges.len(), beacons.len() * (beacons.len() - 1) / 2);
        //dbg!(&edges);

        let n = beacons.len();
        Self { beacons, edges, n }
    }

    fn merge(&mut self, source: &Scanner, transform: &Transform) {
        self.beacons
            .extend(source.beacons.iter().map(|beacon| transform * beacon));
        self.edges.extend(
            source
                .edges
                .iter()
                .map(|(dir, (p0, p1))| (*dir, (transform * p0, transform * p1))),
        );
        self.beacons.sort();
        self.beacons.dedup();
    }

    fn diff_invariant(a: &Position, b: &Position) -> Direction {
        let mut d = a - b;
        let s = d.as_mut_slice();
        for p in s.iter_mut() {
            *p = p.abs();
        }
        s.sort();
        d
    }

    fn try_align(&self, target: &Scanner, transforms: &[Transform]) -> Option<Transform> {
        let corresponding_edges: Vec<_> = self
            .edges
            .iter()
            .filter_map(|(k, v_self)| {
                target
                    .edges
                    .get(k)
                    .map(|v_target| (v_self.clone(), v_target.clone()))
            })
            .collect();
        if corresponding_edges.len() < 12 * (12 - 1) / 2 {
            return None;
        }
        transforms.iter().cloned().find_map(|transform| {
            [false, true].into_iter().find_map(|flip0| {
                let pivot = corresponding_edges[0];
                let reference = pivot.0 .0;
                let anchor = if flip0 { pivot.1 .1 } else { pivot.1 .0 };
                let offset = anchor - transform * reference;
                if corresponding_edges[1..]
                    .iter()
                    .cloned()
                    .all(|((s0, s1), (t0, t1))| {
                        let ts0 = transform * s0 + offset;
                        let ts1 = transform * s1 + offset;
                        (ts0 == t0 && ts1 == t1) || (ts0 == t1 && ts1 == t0)
                    })
                {
                    let mut ret = transform;
                    ret[(0, 3)] = offset[0];
                    ret[(1, 3)] = offset[1];
                    ret[(2, 3)] = offset[2];
                    Some(ret)
                } else {
                    None
                }
            })
        })
    }
}

fn day19(_lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    let mut scanners: Vec<Scanner> = groups
        .iter()
        .rev()
        .map(|g| Scanner::parse(&g[1..]))
        .collect();
    let mut merged = scanners.pop().unwrap();
    let rotations = [
        //
        matrix![1,0,0, 0; 0,1,0, 0; 0,0,1,0; 0,0,0,1],
        matrix![1,0,0, 0; 0,0,1, 0; 0,1,0,0; 0,0,0,1],
        matrix![0,1,0, 0; 1,0,0, 0; 0,0,1,0; 0,0,0,1],
        matrix![0,1,0, 0; 0,0,1, 0; 1,0,0,0; 0,0,0,1],
        matrix![0,0,1, 0; 1,0,0, 0; 0,1,0,0; 0,0,0,1],
        matrix![0,0,1, 0; 0,1,0, 0; 1,0,0,0; 0,0,0,1],
        //
    ];
    let signs = [
        vector![0 + 1, 0 + 1, 0 + 1, 1],
        vector![0 + 1, 0 + 1, 0 - 1, 1],
        vector![0 + 1, 0 - 1, 0 + 1, 1],
        vector![0 + 1, 0 - 1, 0 - 1, 1],
        vector![0 - 1, 0 + 1, 0 + 1, 1],
        vector![0 - 1, 0 + 1, 0 - 1, 1],
        vector![0 - 1, 0 - 1, 0 + 1, 1],
        vector![0 - 1, 0 - 1, 0 - 1, 1],
    ];
    fn determinate3(mat: &Matrix<i16, 4, 4>) -> i16 {
        let dns = mat[(0, 0)] * mat[(1, 1)] * mat[(2, 2)]
            + mat[(0, 1)] * mat[(1, 2)] * mat[(2, 0)]
            + mat[(0, 2)] * mat[(1, 0)] * mat[(2, 1)];
        let ups = mat[(2, 0)] * mat[(1, 1)] * mat[(0, 2)]
            + mat[(2, 1)] * mat[(1, 2)] * mat[(0, 0)]
            + mat[(2, 2)] * mat[(1, 0)] * mat[(0, 1)];
        dns - ups
    }

    let possible_transforms: Vec<_> = rotations
        .iter()
        .flat_map(|rotation| {
            signs.iter().map(|sign| {
                let mut transform = Transform::zero();
                for i in 0..4 {
                    for j in 0..4 {
                        transform[(i, j)] = rotation[(i, j)] * sign[j];
                    }
                }
                transform
            })
        })
        .filter(|t| determinate3(t) > 0)
        .collect();
    let mut transforms = vec![];
    'outer: while !scanners.is_empty() {
        for i in (0..scanners.len()).rev() {
            if let Some(transform) = scanners[i].try_align(&merged, &possible_transforms) {
                merged.merge(&scanners[i], &transform);
                transforms.push(transform);
                scanners.remove(i);
                continue 'outer;
            }
        }
        panic!("Not all overlapping!");
    }

    if gold {
        pairs(transforms.iter())
            .map(|(a, b)| {
                (a - b)
                    .column(3)
                    .as_slice()
                    .iter()
                    .map(|v| v.abs() as usize)
                    .sum()
            })
            .max()
            .unwrap()
    } else {
        merged.beacons.len()
    }
}

fn day20(_lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    let pattern: Vec<bool> = groups[0][0].chars().map(|ch| ch == '#').collect();
    let pattern = &pattern;
    let mut diff_image: Vec<bool> = groups[1]
        .iter()
        .cloned()
        .flat_map(|line: &str| line.chars().map(|ch| ch == '#'))
        .collect();
    let mut width = groups[1][0].chars().count();
    let mut height = diff_image.len() / width;
    let mut rest = false;
    for _step in 1..=(if gold { 50 } else { 2 }) {
        let new_rest = pattern[if rest { 511 } else { 0 }];
        let di = &diff_image;
        diff_image = (0..(height + 2))
            .flat_map(|y| {
                (0..(width + 2)).map(move |x| {
                    let index = self_cross((0..=2).rev())
                    .map(|(sy, sx)| if x >= sx && y >= sy && x-sx < width && y-sy < height {
                            di[(y-sy) * width + x - sx]}else{false}^rest)
                    .fold(0, |n, lit| n * 2 + lit as usize);
                    let lit = pattern[index] ^ new_rest;
                    lit
                })
            })
            .collect();
        width += 2;
        height += 2;
        rest = new_rest;
    }
    diff_image.into_iter().filter(|x| *x).count()
}

fn day21(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let positions: Vec<_> = lines
        .iter()
        .map(|line| {
            scan_fmt!(line, "Player {*d} starting position: {}", usize)
                .ok()
                .unwrap()
        })
        .collect();
    #[derive(PartialEq, Eq, Hash, Default, Clone, Debug)]
    struct WorldState {
        turn: u8,
        positions: [u8; 2],
        scores: [u16; 2],
    }
    let mut quantum_paths = [(0, 0); 7];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let sum = a + b + c;
                quantum_paths[sum].1 = sum + 3;
                quantum_paths[sum].0 += 1;
            }
        }
    }

    let mut universe_wins = [0; 2];
    let mut open_worlds = HashMap::from([(
        WorldState {
            turn: 0,
            positions: [positions[0] as u8, positions[1] as u8],
            scores: [0, 0],
        },
        1,
    )]);
    let winning_score = if gold { 21 } else { 1000 };
    let mut rolls = 0;
    while !open_worlds.is_empty() {
        let mut new_worlds = HashMap::new();
        for (state, count) in open_worlds {
            let deterministic_paths = [(1, (rolls * 3 + 5) % 100 + 1)];
            let dice = if gold {
                &quantum_paths[..]
            } else {
                rolls += 3;
                &deterministic_paths[..]
            };
            for (roll_paths, die_sum) in dice {
                let universes = count * roll_paths;
                let mut state = state.clone();
                let p = state.turn as usize;
                state.positions[p] = (state.positions[p] + *die_sum as u8 - 1) % 10 + 1;
                state.scores[p] += state.positions[p] as u16;
                if state.scores[p] >= winning_score {
                    if !gold {
                        return rolls * state.scores[1 - p] as usize;
                    }
                    universe_wins[p] += universes;
                } else {
                    state.turn = 1 - state.turn;
                    *new_worlds.entry(state).or_insert(0) += universes;
                }
            }
        }
        open_worlds = new_worlds;
    }
    universe_wins.into_iter().max().unwrap()
}

fn day22(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    let commands: Vec<_> = lines
        .iter()
        .flat_map(|line| {
            scan_fmt!(
                line,
                "{/on|off/} x={}..{},y={}..{},z={}..{}",
                String,
                i64,
                i64,
                i64,
                i64,
                i64,
                i64
            )
            .ok()
        })
        // To half-open for clean edge cases
        .map(|(dir, xl, xh, yl, yh, zl, zh)| {
            ([(xl, xh + 1), (yl, yh + 1), (zl, zh + 1)], dir == "on")
        })
        .filter(|(ranges, _)| {
            ranges[..]
                .iter()
                .all(|(l, h)| gold || (*l >= -50 && *h <= 51))
        })
        .collect();

    let mut cuts = [vec![], vec![], vec![]];
    (0..3).for_each(|a| {
        let v = &mut cuts[a];
        *v = commands
            .iter()
            .flat_map(|(ranges, _)| [ranges[a].0, ranges[a].1].into_iter())
            .collect();
        v.sort();
        v.dedup();
    });
    let to_cut = |a: usize, v: i64| cuts[a].binary_search(&v).ok().unwrap();
    let cut_commands = commands
        .into_iter()
        .map(|([(xl, xh), (yl, yh), (zl, zh)], v)| {
            (
                to_cut(0, xl),
                to_cut(0, xh),
                to_cut(1, yl),
                to_cut(1, yh),
                to_cut(2, zl),
                to_cut(2, zh),
                v,
            )
        })
        .collect::<Vec<_>>();
    let [xc, yc, zc] = cuts;
    let mut volume = Array3::from_elem((xc.len(), yc.len(), zc.len()), false);
    for (xl, xh, yl, yh, zl, zh, v) in cut_commands {
        volume.slice_mut(s![xl..xh, yl..yh, zl..zh]).fill(v);
    }

    let mut n = 0;
    for x in 0..(xc.len() - 1) {
        let dx = xc[x + 1] - xc[x];
        for y in 0..(yc.len() - 1) {
            let dy = yc[y] - yc[y];
            for z in 0..(zc.len() - 1) {
                let dz = zc[z + 1] - zc[z];
                if volume[(x, y, z)] {
                    n += dx * dy * dz;
                }
            }
        }
    }

    n as usize
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Pod {
    kind: u8, // 0 = A, 1 = B, 2 = C, 3 = D
    pos: (usize, usize),
}

impl Pod {
    fn step_cost(&self) -> u32 {
        10u32.pow(self.kind as u32)
    }

    fn goal_pos(&self) -> (usize, usize) {
        (2 + self.kind as usize * 2, 2)
    }
}
#[derive(Debug)]
struct Caves {
    pods: Vec<Pod>,
    grid: Array2<Option<usize>>,
}

impl Caves {
    fn new(lines: &[&str]) -> Self {
        let targets = lines
            .iter()
            .flat_map(|line| line.chars())
            .filter(|ch| *ch >= 'A' && *ch <= 'D')
            .map(|ch| ch as usize - 'A' as usize);
        let locations = [
            (2, 1),
            (4, 1),
            (6, 1),
            (8, 1),
            (2, 2),
            (4, 2),
            (6, 2),
            (8, 2),
        ];
        let mut pods: Vec<_> = locations
            .into_iter()
            .zip(targets)
            .map(|(p, t)| Pod {
                pos: p,
                kind: t as u8,
            })
            .collect();
        pods.sort();
        let mut grid = Array2::<Option<usize>>::from_elem((11, 3), None);
        for (i, pod) in pods.iter().enumerate() {
            grid[pod.pos] = Some(i);
        }
        Caves { pods, grid }
    }

    fn d1(a: usize, b: usize) -> usize {
        if a < b {
            b - a
        } else {
            a - b
        }
    }
    fn d2((ax, ay): (usize, usize), (bx, by): (usize, usize)) -> usize {
        Self::d1(ax, bx) + Self::d1(ay, by)
    }
    fn mv(&mut self, i: usize, t: (usize, usize)) -> u32 {
        let pod = &mut self.pods[i];
        let cost = Self::d2(pod.pos, t) as u32 * pod.step_cost();
        let p = &mut pod.pos;
        self.grid[*p] = None;
        *p = t;
        self.grid[*p] = Some(i);
        cost
    }

    fn solve(&mut self) {
        fn step_path<F: FnMut((usize, usize))>(
            (mut x, mut y): (usize, usize),
            (tx, ty): (usize, usize),
            mut f: F,
        ) {
            while y != 0 {
                y -= 1;
                f((x, y));
            }
            if tx > x {
                while x != tx {
                    x += 1;
                    f((x, y));
                }
            } else {
                while x != ty {
                    x -= 1;
                    f((x, y));
                }
            }
            while y != ty {
                y += 1;
                f((x, y));
            }
        }
        fn try_mv(
            caves: &mut Caves,
            i: usize,
            (tx, ty): (usize, usize),
            undo: &mut Vec<(usize, (usize, usize))>,
        ) -> u32 {
            let pod = &caves.pods[i];
            let step_cost = pod.step_cost();
            let mut cost = 0;
            let mut blockers = vec![];
            let (sx, sy) = pod.pos;
            undo.push((i, (sx, sy)));
            step_path((sx, sy), (tx, ty), |p| {
                cost += step_cost;
                if let Some(j) = caves.grid[p] {
                    blockers.push((i, 0));
                }
            });

            let left = min(sx, tx) - 1;
            let right = max(sx, tx) + 1;
            for i in 0..blockers.len() {
                let (bx, _by) = caves.pods[i].pos;
                assert!(bx >= left);
                assert!(bx <= right);
                blockers[i].1 = if bx - left > right - bx { right } else { left }
            }
            for (i, tx) in blockers {
                cost += try_mv(caves, i, (tx, 0), undo);
            }
            cost += caves.mv(i, (tx, ty));
            cost
        }
        while let Some((i, cost)) = (0..self.pods.len())
            .map(|i| {
                let mut undo = vec![];
                let cost = try_mv(self, i, self.pods[i].goal_pos(), &mut undo);
                self.print();
                //for (i, p) in undo {}
                (i, cost)
            })
            .min_by_key(|(_, cost)| *cost)
        {
            self.print();
            panic!();
        }
    }
    fn ch(&self, value: Option<usize>) -> char {
        match value {
            None => '.',
            Some(i) => ('A' as u8 + self.pods[i].kind as u8) as char,
        }
    }
    fn print(&self) {
        eprintln!("#############");
        for y in 0..3 {
            eprint!("#");
            for x in 0..11 {
                eprint!(
                    "{}",
                    if y > 0 && ![2, 4, 6, 8].contains(&x) {
                        '#'
                    } else {
                        self.ch(self.grid[(x, 0)])
                    }
                );
            }
            eprintln!("#");
        }
        eprintln!("#############");
    }
}

fn day23(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    if gold {
        return 0;
    }
    let mut caves = Caves::new(lines);
    caves.solve();

    dbg!(&caves);
    1
}

fn day24(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
    if gold {
        return 0;
    }
    #[derive(Debug)]
    enum Var {
        W,
        X,
        Y,
        Z,
    }
    #[derive(Debug)]
    enum Arg {
        Var(Var),
        Lit(i64),
    }
    #[derive(Debug)]
    enum Code {
        Input(Var),
        Add(Var, Arg),
        Mul(Var, Arg),
        Div(Var, Arg),
        Mod(Var, Arg),
        Eql(Var, Arg),
    }
    let parse_var = |s: &str| match s {
        "w" => Var::W,
        "x" => Var::X,
        "y" => Var::Y,
        "z" => Var::Z,
        v => panic!("{} isn't a var", v),
    };
    let parse_arg = |s: &str| {
        if s.chars().last().unwrap().is_numeric() {
            Arg::Lit(s.parse().ok().unwrap())
        } else {
            Arg::Var(parse_var(s))
        }
    };

    let program: Vec<_> = lines
        .iter()
        .map(|line| {
            let mut words = line.split(' ');
            match words.next().unwrap() {
                "inp" => Code::Input(parse_var(words.next().unwrap())),
                "add" => Code::Add(
                    parse_var(words.next().unwrap()),
                    parse_arg(words.next().unwrap()),
                ),
                "mul" => Code::Mul(
                    parse_var(words.next().unwrap()),
                    parse_arg(words.next().unwrap()),
                ),
                "div" => Code::Div(
                    parse_var(words.next().unwrap()),
                    parse_arg(words.next().unwrap()),
                ),
                "mod" => Code::Mod(
                    parse_var(words.next().unwrap()),
                    parse_arg(words.next().unwrap()),
                ),
                "eql" => Code::Eql(
                    parse_var(words.next().unwrap()),
                    parse_arg(words.next().unwrap()),
                ),
                o => panic!("Unrecognized: {}", o),
            }
        })
        .collect();
    dbg!(&program);
    #[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
    enum Expression {
        Input(usize),
        Literal(i64),
        Add(Box<(Expression, Expression)>),
        Mul(Box<(Expression, Expression)>),
        Div(Box<(Expression, Expression)>),
        Mod(Box<(Expression, Expression)>),
        Eql(Box<(Expression, Expression)>),
        Ne(Box<(Expression, Expression)>),
    };
    fn print2(op: &str, p: &Box<(Expression, Expression)>, depth: usize) {
        if depth == 0 {
            eprint!("...");
            return;
        }
        eprint!("(");
        print(&(*p).0, depth - 1);
        eprint!("{}", op);
        print(&(*p).1, depth - 1);
        eprint!(")");
    }

    fn print(e: &Expression, depth: usize) {
        /*
        ((1+(25*(0=([13]=(-2+(((...*...)+(...*...))%26))))))*((((1+(25*(0=([12]=(-16+...)))))*((((1+(25*...))*((...*...)+(...*...)))+((10+[11])*(0=([11]=...))))/26))+((8+[12])*(0=([12]=(-16+((...+...)%26))))))/26))
        */
        match e {
            Expression::Literal(v) => eprint!("{}", v),
            Expression::Input(n) => eprint!("i[{}]", n),
            Expression::Add(p) => print2("+", p, depth),
            Expression::Mul(p) => print2("*", p, depth),
            Expression::Div(p) => print2("/", p, depth),
            Expression::Mod(p) => print2("%", p, depth),
            Expression::Eql(p) => print2("==", p, depth),
            Expression::Ne(p) => print2("!=", p, depth),
            _ => eprint!("??"),
        }
    }
    let mut vars = [
        Expression::Literal(0),
        Expression::Literal(0),
        Expression::Literal(0),
        Expression::Literal(0),
    ];
    let mut inputs = 0;
    let mut last_inputs = 0;
    for code in program {
        let pv = |var: &Var| match var {
            Var::W => 0,
            Var::X => 1,
            Var::Y => 2,
            Var::Z => 3,
        };
        match code {
            Code::Input(var) => {
                vars[pv(&var)] = {
                    inputs += 1;
                    Expression::Input(inputs - 1)
                };
            }
            Code::Add(var, arg) => {
                vars[pv(&var)] = Expression::Add(Box::new((
                    vars[pv(&var)].clone(),
                    match arg {
                        Arg::Lit(v) => Expression::Literal(v),
                        Arg::Var(var) => vars[pv(&var)].clone(),
                    },
                )));
            }
            Code::Mul(var, arg) => {
                vars[pv(&var)] = Expression::Mul(Box::new((
                    vars[pv(&var)].clone(),
                    match arg {
                        Arg::Lit(v) => Expression::Literal(v),
                        Arg::Var(var) => vars[pv(&var)].clone(),
                    },
                )));
            }
            Code::Div(var, arg) => {
                vars[pv(&var)] = Expression::Div(Box::new((
                    vars[pv(&var)].clone(),
                    match arg {
                        Arg::Lit(v) => Expression::Literal(v),
                        Arg::Var(var) => vars[pv(&var)].clone(),
                    },
                )));
            }
            Code::Mod(var, arg) => {
                vars[pv(&var)] = Expression::Mod(Box::new((
                    vars[pv(&var)].clone(),
                    match arg {
                        Arg::Lit(v) => Expression::Literal(v),
                        Arg::Var(var) => vars[pv(&var)].clone(),
                    },
                )));
            }
            Code::Eql(var, arg) => {
                vars[pv(&var)] = Expression::Eql(Box::new((
                    vars[pv(&var)].clone(),
                    match arg {
                        Arg::Lit(v) => Expression::Literal(v),
                        Arg::Var(var) => vars[pv(&var)].clone(),
                    },
                )));
            }
        };
        fn ge(e: &Expression, min: i64) -> bool {
            match e {
                Expression::Input(_) => min <= 1,
                Expression::Literal(v) => *v >= min,
                Expression::Add(p) => match p.as_ref() {
                    (Expression::Input(_), b) => ge(b, min - 1),
                    (Expression::Literal(v), b) => ge(b, min - v),
                    _ => false,
                },
                Expression::Mod(p) => match p.as_ref() {
                    (a, Expression::Literal(d)) => min <= 0 && ge(a, 0),
                    (a, Expression::Literal(_)) => false,
                    _ => panic!(),
                },
                _ => false,
            }
        }
        fn le(e: &Expression, max: i64) -> bool {
            match e {
                Expression::Input(_) => max >= 10,
                Expression::Literal(v) => *v <= max,
                Expression::Add(p) => match p.as_ref() {
                    (o, Expression::Literal(n)) => le(o, max - *n),
                    (Expression::Literal(n), o) => le(o, max - *n),
                    _ => false,
                },
                _ => false,
            }
        }
        fn simplify(e: &mut Expression) -> bool {
            match e {
                Expression::Input(_) => false,
                Expression::Literal(_) => false,
                Expression::Add(p) => match p.as_mut() {
                    (a, b) if *a > *b => {
                        *p.as_mut() = (b.clone(), a.clone());
                        true
                    }
                    (Expression::Literal(a), Expression::Literal(b)) => {
                        *e = Expression::Literal(*a + *b);
                        eprintln!("const+");
                        true
                    }
                    (Expression::Literal(0), o) => {
                        *e = o.clone();
                        eprintln!("0+");
                        true
                    }
                    (o, Expression::Literal(0)) => {
                        *e = o.clone();
                        eprintln!("+0");
                        true
                    }
                    (Expression::Literal(a), Expression::Add(p))
                        if match p.as_ref() {
                            (Expression::Literal(_), _) => true,
                            (x, Expression::Literal(_)) => true,
                            _ => false,
                        } =>
                    {
                        let (s, x) = match p.as_ref() {
                            (Expression::Literal(b), x) => (*a + *b, x.clone()),
                            (x, Expression::Literal(b)) => (*a + *b, x.clone()),
                            _ => unreachable!(),
                        };
                        *e = Expression::Add(Box::new((Expression::Literal(s), x)));
                        eprintln!("a+b+x");
                        true
                    }

                    (a, b) => simplify(a) || simplify(b),
                },
                Expression::Mul(p) => match p.as_mut() {
                    (a, b) if *a > *b => {
                        *p.as_mut() = (b.clone(), a.clone());
                        eprintln!("flip*");
                        true
                    }
                    (Expression::Literal(a), Expression::Literal(b)) => {
                        *e = Expression::Literal(*a * *b);
                        eprintln!("const*");
                        true
                    }
                    (Expression::Literal(0), _) => {
                        *e = Expression::Literal(0);
                        eprintln!("*0");
                        true
                    }
                    (Expression::Literal(1), o) => {
                        *e = o.clone();
                        eprintln!("*1");
                        true
                    }
                    (a, b) => simplify(a) || simplify(b),
                },
                Expression::Div(p) => match p.as_mut() {
                    (Expression::Literal(a), Expression::Literal(b)) => {
                        *e = Expression::Literal(*a / *b);
                        eprintln!("const/");
                        true
                    }
                    (o, Expression::Literal(1)) => {
                        *e = o.clone();
                        eprintln!("/1");
                        true
                    }
                    (Expression::Add(p), Expression::Literal(d))
                        if match p.as_ref() {
                            (_, Expression::Mul(p))
                                if match p.as_ref() {
                                    (Expression::Literal(m), _) if m == d => true,
                                    _ => false,
                                } =>
                            {
                                true
                            }
                            _ => false,
                        } =>
                    {
                        *e = match p.as_ref() {
                            (_, Expression::Mul(p)) => match p.as_ref() {
                                (Expression::Literal(m), r) => r,
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        }
                        .clone();
                        eprintln!("(small+M*b)/M");
                        true
                    }
                    (a, b) => simplify(a) || simplify(b),
                },
                Expression::Mod(p) => match p.as_mut() {
                    (Expression::Literal(a), Expression::Literal(b)) => {
                        *e = Expression::Literal(*a % *b);
                        eprintln!("const%");
                        true
                    }
                    (left, Expression::Literal(m)) if le(left, *m - 1) => {
                        *e = left.clone();
                        eprintln!("small%big");
                        true
                    }
                    (Expression::Add(p), Expression::Literal(m))
                        if match p.as_ref() {
                            (_, Expression::Mul(p))
                                if match p.as_ref() {
                                    (Expression::Literal(s), _) if s == m => true,
                                    _ => false,
                                } =>
                            {
                                true
                            }
                            _ => false,
                        } =>
                    {
                        //(a+b*c)%b
                        //Expression::=> HHHHHHHHHHHHHKX) = e => {
                        *e = Expression::Mod(Box::new((
                            p.as_ref().0.clone(),
                            Expression::Literal(*m),
                        )));
                        eprintln!("(a+M*b)%M");
                        true
                    }
                    (Expression::Literal(0), _) => {
                        *e = Expression::Literal(0);
                        eprintln!("0%");
                        true
                    }
                    (a, b) => simplify(a) || simplify(b),
                },
                Expression::Eql(p) => match p.as_mut() {
                    (a, b) if *a > *b => {
                        *p.as_mut() = (b.clone(), a.clone());
                        eprintln!("flip=");
                        true
                    }
                    (Expression::Literal(a), Expression::Literal(b)) => {
                        *e = Expression::Literal(if *a == *b { 1 } else { 0 });
                        eprintln!("const=");
                        true
                    }
                    (Expression::Input(_), o) if le(o, 0) => {
                        print(o, 0);
                        eprintln!("\n^< 1");
                        *e = Expression::Literal(0);
                        true
                    }
                    (Expression::Input(_), o) if ge(o, 10) => {
                        print(o, 0);
                        eprintln!("\n^> 9");
                        *e = Expression::Literal(0);
                        true
                    }
                    (Expression::Literal(0), Expression::Eql(p)) => {
                        *e = Expression::Ne(p.clone());
                        eprintln!("!=%");
                        true
                    }
                    (a, b) => simplify(a) || simplify(b),
                },
                Expression::Ne(p) => match p.as_mut() {
                    (a, b) => simplify(a) || simplify(b),
                },
            }
        }
        let before = vars[3].clone();
        for i in 0..4 {
            let v = &mut vars[i];
            let before = v.clone();
            while simplify(v) {}
            if before != *v && false {
                eprint!("from ");
                print(&before, 0);
                eprint!("\n to ");
                print(v, 0);
                eprintln!();
                eprintln!("\n-------------------------------");
            }
        }
        if inputs > 7 && false {
            break;
        }
        if inputs != last_inputs {
            last_inputs = inputs;
            eprint!("z = ");
            print(&before, 20);
            eprintln!("\n now {} inputs", inputs);
        }
    }
    let z = &vars[3];
    fn apply((a, b): &(Expression, Expression), vars: &[i64], op: &dyn Fn(i64, i64) -> i64) -> i64 {
        let a = eval(a, vars);
        let b = eval(b, vars);
        op(a, b)
    }
    fn eval(e: &Expression, vars: &[i64]) -> i64 {
        match e {
            Expression::Literal(v) => *v,
            Expression::Input(n) => vars[*n],
            Expression::Add(p) => apply(p, vars, &|a, b| a + b),
            Expression::Mul(p) => apply(p, vars, &|a, b| a * b),
            Expression::Div(p) => apply(p, vars, &|a, b| a / b),
            Expression::Mod(p) => apply(p, vars, &|a, b| a % b),
            Expression::Eql(p) => apply(p, vars, &|a, b| (a == b) as i64),
            Expression::Ne(p) => apply(p, vars, &|a, b| (a != b) as i64),
        }
    }
    print(z, 10);
    print(z, 20);
    let mut inputs = [1; 14];
    for position in 0..inputs.len() {
        for v in 1..=9 {
            inputs[position] = v;
            dbg!(eval(z, &inputs));
        }
    }

    0
}

fn day25(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

#[derive(Parser)]
#[clap(version = "1.0", author = "Tom Jackson <ddrcoder@gmail.com>")]
struct Opts {
    problems: Vec<usize>,
}

fn main() {
    fn content(filename: &str) -> Option<String> {
        let mut r = String::new();
        File::open(filename).ok()?.read_to_string(&mut r).ok()?;
        Some(r)
    }

    let opts = Opts::parse();
    let solutions = [
        day1, day2, day3, day4, day5, day6, day7, day8, day9, day10, day11, day12, day13, day14,
        day15, day16, day17, day18, day19, day20, day21, day22, day23, day24, day25,
    ];

    let wide = 20;
    println!(
        "{1:>2}: {2:>0$} {3:>0$}\n    {4:>0$} {5:>0$}",
        wide + 13,
        "#",
        "test_silver",
        "test_gold",
        "silver",
        "gold"
    );
    for (i, solution) in solutions.iter().enumerate() {
        let n = i + 1;
        if !opts.problems.is_empty() && !opts.problems.contains(&n) {
            continue;
        }
        for test in [true, false] {
            if test {
                print!("{:2}:", n);
            } else {
                print!("\n   ");
            }
            if let Some(content) = content(&format!(
                "input/day{}{}.txt",
                n,
                if test { "t" } else { "" }
            )) {
                let lines: Vec<_> = content.split('\n').collect();
                let lines = if lines[lines.len() - 1].is_empty() {
                    &lines[0..(lines.len() - 1)]
                } else {
                    &lines
                };
                let groups: Vec<_> = lines
                    .split(|s| s.is_empty())
                    .filter(|g| !g.is_empty())
                    .collect();
                let timed = |f: &mut dyn FnMut() -> usize| {
                    let start = Instant::now();
                    let ret = f();
                    (start.elapsed(), ret)
                };

                for gold in [false, true] {
                    let (time, value) = timed(&mut || solution(lines, &groups, gold));

                    print!(" {0:1$} ({2:8}us)", value, wide, time.as_micros());
                }
            } else {
                for _gold in [false, true] {
                    print!(" {0:>1$} ({2:>8}us)", "-", wide, "-");
                }
            }
        }
        println!();
    }
}
