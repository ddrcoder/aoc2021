#[macro_use]
extern crate scan_fmt;
extern crate clap;

use clap::Parser;
use std::str::FromStr;
//use std::hash::Hash;
use std::{collections::VecDeque, fmt::Debug};

use regex::{Captures, Regex};
//use std::collections::{hash_map::HashMap, hash_set::HashSet};
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

fn day3(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day4(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day5(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day6(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day7(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day8(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
}

fn day9(_lines: &[&str], _groups: &[&[&str]], _gold: bool) -> usize {
    0
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
