use criterion::{Criterion, criterion_group, criterion_main};
use rust_workshop_2::{Histogram, integrate};

fn benchmark_load(c: &mut Criterion) {
    c.bench_function("load", |b| {
        b.iter(|| {
            let _ = Histogram::load("histogram.h5").unwrap();
        })
    });
}

fn benchmark_integrate(c: &mut Criterion) {
    let hist = Histogram::load("histogram.h5").unwrap();
    c.bench_function("integrate", |b| {
        b.iter(|| {
            let _ = integrate(hist.clone(), (-1.0, 6.0));
        })
    });
}

criterion_group!(benches, benchmark_load, benchmark_integrate);
criterion_main!(benches);
