use criterion::{Criterion, criterion_group, criterion_main};
use piv_core::{PivAnalyzer, PivConfig, PreparedFrame, PreprocessConfig, synthetic};

fn full_hd_pair(c: &mut Criterion) {
    let (a, b) = synthetic::fog_pair(1080, 1920, 1, |x, y| (2.0 + 0.001 * y, -1.0 + 0.001 * x));
    let pre = PreprocessConfig::default();

    c.bench_function("prepare 1920x1080 frame", |bench| {
        bench.iter(|| PreparedFrame::new(&a, &pre, None))
    });

    let (pa, pb) = (
        PreparedFrame::new(&a, &pre, None),
        PreparedFrame::new(&b, &pre, None),
    );
    let analyzer = PivAnalyzer::new(PivConfig::default()).unwrap(); // 64 → 32 px, 50 % overlap
    c.bench_function("analyze 1920x1080 pair (64,32 / 50%)", |bench| {
        bench.iter(|| analyzer.analyze(&pa, &pb).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = full_hd_pair
}
criterion_main!(benches);
