#[macro_use]
extern crate scan_fmt;
extern crate clap;

use clap::Parser;
use std::cmp::max;
use std::cmp::min;
use std::str::FromStr;
//use std::hash::Hash;
use std::{collections::VecDeque, fmt::Debug};

use regex::{Captures, Regex};
use std::collections::{hash_map::HashMap, hash_set::HashSet};
use std::fs::File;
use std::io::Read;
//use std::iter::{once, Iterator};

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
    let mut grid: HashMap<(i64, i64), i64> = HashMap::new();
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
            *grid.entry((x, y)).or_insert(0) += 1;
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
    return 0;
    let crabs: Vec<i64> = lines[0].split(',').flat_map(|n| n.parse().ok()).collect();
    let (max, min) = (crabs.iter().max(), crabs.iter().min());
    (*min.unwrap()..=*max.unwrap())
        .map(|t| {
            crabs
                .iter()
                .map(|p| (t - p).abs() as usize)
                .map(|d| if gold { (d * d + d) / 2 } else { d })
                .sum()
        })
        .min()
        .unwrap()
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

fn day9(lines: &[&str], _groups: &[&[&str]], gold: bool) -> usize {
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

fn day10(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day11(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day12(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day13(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}
fn day14(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day15(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day16(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day17(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day18(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day19(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}
fn day20(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day21(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day22(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day23(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day24(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
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
        "{1:>2}: {2:>0$} {3:>0$} {4:>0$} {5:>0$}",
        wide, "#", "test_silver", "test_gold", "silver", "gold"
    );
    for (i, solution) in solutions.iter().enumerate() {
        let n = i + 1;
        if !opts.problems.is_empty() && !opts.problems.contains(&n) {
            continue;
        }
        print!("{:2}:", n);
        for test in [true, false] {
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
                for gold in [false, true] {
                    print!(" {0:1$}", solution(lines, &groups, gold), wide);
                }
            } else {
                for _gold in [false, true] {
                    print!(" {0:>1$}", "-", wide);
                }
            }
        }
        println!();
    }
}
