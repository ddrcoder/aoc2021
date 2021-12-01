#[macro_use]
extern crate scan_fmt;
extern crate clap;

use clap::Parser;
use std::hash::Hash;
use std::{collections::VecDeque, fmt::Debug};

use regex::Regex;
use std::collections::{hash_map::HashMap, hash_set::HashSet};
use std::fs::File;
use std::io::Read;
use std::iter::{once, Iterator};

fn day1(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
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

fn day2(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day3(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day4(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day5(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day6(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day7(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day8(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day9(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day10(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day11(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day12(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day13(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}
fn day14(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day15(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day16(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day17(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day18(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day19(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}
fn day20(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day21(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day22(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day23(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day24(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
    0
}

fn day25(lines: &[&str], groups: &[&[&str]], gold: bool) -> usize {
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
                let groups: Vec<_> = lines
                    .split(|s| s.is_empty())
                    .filter(|g| !g.is_empty())
                    .collect();
                for gold in [false, true] {
                    print!(" {0:1$}", solution(&lines, &groups, gold), wide);
                }
            } else {
                for gold in [false, true] {
                    print!(" {0:>1$}", "-", wide);
                }
            }
        }
        println!();
    }
}
