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
    for windows in [vec![96, 64], vec![64, 32]] {
        let name = format!("analyze 1920x1080 pair ({windows:?}, 50% overlap)");
        let analyzer = PivAnalyzer::new(PivConfig {
            windows,
            ..Default::default()
        })
        .unwrap();
        c.bench_function(&name, |bench| {
            bench.iter(|| analyzer.analyze(&pa, &pb).unwrap())
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = full_hd_pair
}
criterion_main!(benches);
