use fastrand;
use fxhash::FxHashMap;
use statrs::distribution::DiscreteCDF;
use statrs::distribution::Poisson;
use sylph::cmdline::ContainArgs;
use sylph::constants::CUTOFF_PVALUE;
use sylph::constants::MAX_MEDIAN_FOR_MEAN_FINAL_EST;
use sylph::constants::MEDIAN_ANI_THRESHOLD;
use sylph::constants::MED_KMER_FOR_ID_EST;
use sylph::constants::MIN_ANI_DEF;
use sylph::constants::MIN_ANI_P_DEF;
use sylph::inference::binary_search_lambda;
use sylph::inference::mean;
use sylph::inference::mle_zip;
use sylph::inference::mme_lambda;
use sylph::inference::ratio_lambda;
use sylph::inference::var;
use sylph::types::AdjustStatus;
use sylph::types::AniResult;
use sylph::types::GenomeSketch;
use sylph::types::Kmer;
use sylph::types::SequencesSketch;

pub fn derep_if_reassign_threshold<'a>(
    results_old: &Vec<AniResult>,
    results_new: Vec<AniResult<'a>>,
    ani_thresh: f64,
    k: usize,
) -> Vec<AniResult<'a>> {
    let ani_thresh = ani_thresh / 100.;

    let mut gn_sketch_to_contain = FxHashMap::default();
    for result in results_old.iter() {
        gn_sketch_to_contain.insert(result.genome_sketch, result);
    }

    let threshold = f64::powf(ani_thresh, k as f64);
    let mut return_vec = vec![];
    for result in results_new.into_iter() {
        let old_res = &gn_sketch_to_contain[result.genome_sketch];
        let num_kmer_reassign = (old_res.containment_index.0 - result.containment_index.0) as f64;
        let reass_thresh = threshold * result.containment_index.1 as f64;
        if num_kmer_reassign < reass_thresh {
            return_vec.push(result);
        }
        // else{
        //     log::debug!("genome {} had num k-mers reassigned = {}, threshold was {}, removing.", result.gn_name, num_kmer_reassign, reass_thresh);
        // }
    }
    return return_vec;
}

pub fn estimate_true_cov(
    results: &mut Vec<AniResult>,
    kmer_id_opt: Option<f64>,
    estimate_unknown: bool,
    read_length: f64,
    k: usize,
) {
    let mut multiplier = 1.;
    if estimate_unknown {
        multiplier = read_length / (read_length - k as f64 + 1.);
    }
    if estimate_unknown && kmer_id_opt.is_some() {
        let id = kmer_id_opt.unwrap();
        for res in results.iter_mut() {
            res.final_est_cov = res.final_est_cov / id * multiplier;
        }
    }
}

pub fn estimate_covered_bases(
    results: &Vec<AniResult>,
    sequence_sketch: &SequencesSketch,
    read_length: f64,
    k: usize,
) -> f64 {
    let multiplier = read_length / (read_length - (k as f64) + 1.);

    let mut num_covered_bases = 0.;
    for res in results.iter() {
        num_covered_bases += (res.genome_sketch.gn_size as f64) * res.final_est_cov
    }
    let mut num_total_counts = 0;
    for count in sequence_sketch.kmer_counts.values() {
        num_total_counts += *count as usize;
    }
    let num_tentative_bases = sequence_sketch.c * num_total_counts;
    let num_tentative_bases = num_tentative_bases as f64 * multiplier;
    if num_tentative_bases == 0. {
        return 0.;
    }
    return f64::min(num_covered_bases as f64 / num_tentative_bases, 1.);
}

pub fn winner_table<'a>(
    results: &'a Vec<AniResult>,
    _log_reassign: bool,
) -> FxHashMap<Kmer, (f64, &'a GenomeSketch, bool)> {
    let mut kmer_to_genome_map: FxHashMap<_, _> = FxHashMap::default();
    for res in results.iter() {
        //let gn_sketch = &genome_sketches[res.genome_sketch_index];
        let gn_sketch = res.genome_sketch;
        for kmer in gn_sketch.genome_kmers.iter() {
            let v = kmer_to_genome_map.entry(*kmer).or_insert((
                res.final_est_ani,
                res.genome_sketch,
                false,
            ));
            if res.final_est_ani > v.0 {
                *v = (res.final_est_ani, gn_sketch, true);
            }
        }

        if gn_sketch.pseudotax_tracked_nonused_kmers.is_some() {
            for kmer in gn_sketch
                .pseudotax_tracked_nonused_kmers
                .as_ref()
                .unwrap()
                .iter()
            {
                let v = kmer_to_genome_map.entry(*kmer).or_insert((
                    res.final_est_ani,
                    res.genome_sketch,
                    false,
                ));
                if res.final_est_ani > v.0 {
                    *v = (res.final_est_ani, gn_sketch, true);
                }
            }
        }
    }

    //log reassigned kmers
    // if log_reassign{
    //     log::info!("------------- Logging k-mer reassignments -----------------");
    //     let mut sketch_to_index = FxHashMap::default();
    //     for (i,res) in results.iter().enumerate(){
    //         log::info!("Index\t{}\t{}\t{}", i, res.genome_sketch.file_name, res.genome_sketch.first_contig_name);
    //         sketch_to_index.insert(res.genome_sketch, i);
    //     }
    //     (0..results.len()).into_par_iter().for_each(|i|{
    //         let res = &results[i];
    //         let mut reassign_edge_map = FxHashMap::default();
    //         for kmer in res.genome_sketch.genome_kmers.iter(){
    //             let value = kmer_to_genome_map[kmer].1;
    //             if value != res.genome_sketch{
    //                 let edge_count = reassign_edge_map.entry((sketch_to_index[value],i)).or_insert(0);
    //                 *edge_count += 1;
    //             }
    //         }
    //         for (key,val) in reassign_edge_map{
    //             if val > 10{
    //                 log::info!("{}->{}\t{}\tkmers reassigned", key.0, key.1, val);
    //             }
    //         }
    //     });
    // }

    return kmer_to_genome_map;
}

pub fn get_stats<'a>(
    args: &ContainArgs,
    genome_sketch: &'a GenomeSketch,
    sequence_sketch: &SequencesSketch,
    winner_map: Option<&FxHashMap<Kmer, (f64, &GenomeSketch, bool)>>,
    log_reassign: bool,
) -> Option<AniResult<'a>> {
    if genome_sketch.k != sequence_sketch.k {
        // log::error!(
        //     "k parameter for reads {} != k parameter for genome {}",
        //     sequence_sketch.k,
        //     genome_sketch.k
        // );
        std::process::exit(1);
    }
    if genome_sketch.c < sequence_sketch.c {
        // log::error!(
        //     "c parameter for reads {} > c parameter for genome {}",
        //     sequence_sketch.c,
        //     genome_sketch.c
        // );
        std::process::exit(1);
    }
    let mut contain_count = 0;
    let mut covs = vec![];
    let gn_kmers = &genome_sketch.genome_kmers;
    if (gn_kmers.len() as f64) < args.min_number_kmers {
        return None;
    }

    let mut kmers_lost_count = 0;
    for kmer in gn_kmers.iter() {
        if sequence_sketch.kmer_counts.contains_key(kmer) {
            if sequence_sketch.kmer_counts[kmer] == 0 {
                continue;
            }
            if winner_map.is_some() {
                let map = &winner_map.unwrap();
                if map[kmer].1 != genome_sketch {
                    kmers_lost_count += 1;
                    continue;
                }
                contain_count += 1;
                covs.push(sequence_sketch.kmer_counts[kmer]);
            } else {
                contain_count += 1;
                covs.push(sequence_sketch.kmer_counts[kmer]);
            }
        }
    }

    if covs.is_empty() {
        return None;
    }
    let naive_ani = f64::powf(
        contain_count as f64 / gn_kmers.len() as f64,
        1. / genome_sketch.k as f64,
    );
    covs.sort();
    //let covs = &covs[0..covs.len() * 99 / 100];
    let median_cov = covs[covs.len() / 2] as f64;
    let pois = Poisson::new(median_cov).unwrap();
    let mut max_cov = f64::MAX;
    if median_cov < 30. {
        for i in covs.len() / 2..covs.len() {
            let cov = covs[i];
            if pois.cdf(cov.into()) < CUTOFF_PVALUE {
                max_cov = cov as f64;
            } else {
                break;
            }
        }
    }
    // log::trace!("COV VECTOR for {}/{}: {:?}, MAX_COV_THRESHOLD: {}", sequence_sketch.file_name, genome_sketch.file_name ,covs, max_cov);

    let mut full_covs = vec![0; gn_kmers.len() - contain_count];
    for cov in covs.iter() {
        if (*cov as f64) <= max_cov {
            full_covs.push(*cov);
        }
    }
    let var = var(&full_covs);
    if var.is_some() {
        // log::trace!("VAR {} {}", var.unwrap(), genome_sketch.file_name);
    }
    let mean_cov = full_covs.iter().sum::<u32>() as f64 / full_covs.len() as f64;
    let geq1_mean_cov = full_covs.iter().sum::<u32>() as f64 / covs.len() as f64;

    let use_lambda;
    if median_cov > MEDIAN_ANI_THRESHOLD {
        use_lambda = AdjustStatus::High
    } else {
        let test_lambda;
        if args.ratio {
            test_lambda = ratio_lambda(&full_covs, args.min_count_correct)
        } else if args.mme {
            test_lambda = mme_lambda(&full_covs)
        } else if args.nb {
            test_lambda = binary_search_lambda(&full_covs)
        } else if args.mle {
            test_lambda = mle_zip(&full_covs, sequence_sketch.k as f64)
        } else {
            test_lambda = ratio_lambda(&full_covs, args.min_count_correct)
        };
        if test_lambda.is_none() {
            use_lambda = AdjustStatus::Low
        } else {
            use_lambda = AdjustStatus::Lambda(test_lambda.unwrap());
        }
    }

    let final_est_cov;

    if let AdjustStatus::Lambda(lam) = use_lambda {
        final_est_cov = lam
    } else if median_cov < MAX_MEDIAN_FOR_MEAN_FINAL_EST {
        final_est_cov = geq1_mean_cov;
    } else {
        if args.mean_coverage {
            final_est_cov = geq1_mean_cov;
        } else {
            final_est_cov = median_cov;
        }
    }

    let opt_lambda;
    if use_lambda == AdjustStatus::Low || use_lambda == AdjustStatus::High {
        opt_lambda = None
    } else {
        opt_lambda = Some(final_est_cov)
    };

    let opt_est_ani = ani_from_lambda(opt_lambda, mean_cov, sequence_sketch.k as f64, &full_covs);

    let final_est_ani;
    if opt_lambda.is_none() || opt_est_ani.is_none() || args.no_adj {
        final_est_ani = naive_ani;
    } else {
        final_est_ani = opt_est_ani.unwrap();
    }

    let min_ani = if args.minimum_ani.is_some() {
        args.minimum_ani.unwrap() / 100.
    } else if args.pseudotax {
        MIN_ANI_P_DEF
    } else {
        MIN_ANI_DEF
    };
    if final_est_ani < min_ani {
        if winner_map.is_some() {
            //Used to be > min ani, now it is not after reassignment
            if log_reassign {
                // log::info!("Genome/contig {}/{} has ANI = {} < {} after reassigning {} k-mers ({} contained k-mers after reassign)",
                //     genome_sketch.file_name,
                //     genome_sketch.first_contig_name,
                //     final_est_ani * 100.,
                //     min_ani * 100.,
                //     kmers_lost_count,
                //     contain_count)
            }
        }
        return None;
    }

    let (mut low_ani, mut high_ani, mut low_lambda, mut high_lambda) = (None, None, None, None);
    if !args.no_ci && opt_lambda.is_some() {
        let bootstrap = bootstrap_interval(&full_covs, sequence_sketch.k as f64, &args);
        low_ani = bootstrap.0;
        high_ani = bootstrap.1;
        low_lambda = bootstrap.2;
        high_lambda = bootstrap.3;
    }

    let seq_name;
    if let Some(sample) = &sequence_sketch.sample_name {
        seq_name = sample.clone();
    } else {
        seq_name = sequence_sketch.file_name.clone();
    }

    let kmers_lost;
    if winner_map.is_some() {
        kmers_lost = Some(kmers_lost_count)
    } else {
        kmers_lost = None;
    }

    let ani_result = AniResult {
        naive_ani,
        final_est_ani,
        final_est_cov,
        seq_name: seq_name,
        gn_name: genome_sketch.file_name.as_str(),
        contig_name: genome_sketch.first_contig_name.as_str(),
        mean_cov: geq1_mean_cov,
        median_cov,
        containment_index: (contain_count, gn_kmers.len()),
        lambda: use_lambda,
        ani_ci: (low_ani, high_ani),
        lambda_ci: (low_lambda, high_lambda),
        genome_sketch: genome_sketch,
        rel_abund: None,
        seq_abund: None,
        kmers_lost: kmers_lost,
    };
    //log::trace!("Other time {:?}", Instant::now() - start_t_initial);

    return Some(ani_result);
}

fn ani_from_lambda(lambda: Option<f64>, _mean: f64, k: f64, full_cov: &[u32]) -> Option<f64> {
    if lambda.is_none() {
        return None;
    }
    let mut contain_count = 0;
    let mut _zero_count = 0;
    for x in full_cov {
        if *x != 0 {
            contain_count += 1;
        } else {
            _zero_count += 1;
        }
    }

    let lambda = lambda.unwrap();
    let adj_index =
        contain_count as f64 / (1. - f64::powf(2.78281828, -lambda)) / full_cov.len() as f64;
    let ret_ani;
    //let ani = f64::powf(1. - pi, 1./k);
    let ani = f64::powf(adj_index, 1. / k);
    if ani < 0. || ani.is_nan() {
        ret_ani = None;
    } else {
        if ani > 1. {
            ret_ani = Some(ani)
        } else {
            ret_ani = Some(ani);
        }
    }
    return ret_ani;
}

fn bootstrap_interval(
    covs_full: &Vec<u32>,
    k: f64,
    args: &ContainArgs,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    fastrand::seed(7);
    let num_samp = covs_full.len();
    let iters = 100;
    let mut res_ani = vec![];
    let mut res_lambda = vec![];

    for _ in 0..iters {
        let mut rand_vec = vec![];
        rand_vec.reserve(num_samp);
        for _ in 0..num_samp {
            rand_vec.push(covs_full[fastrand::usize(..covs_full.len())]);
        }
        let lambda;
        if args.ratio {
            lambda = ratio_lambda(&rand_vec, args.min_count_correct);
        } else if args.mme {
            lambda = mme_lambda(&rand_vec);
        } else if args.nb {
            lambda = binary_search_lambda(&rand_vec);
        } else if args.mle {
            lambda = mle_zip(&rand_vec, k);
        } else {
            lambda = ratio_lambda(&rand_vec, args.min_count_correct);
        }
        let ani = ani_from_lambda(lambda, mean(&rand_vec).unwrap().into(), k, &rand_vec);
        if ani.is_some() && lambda.is_some() {
            if !ani.unwrap().is_nan() && !lambda.unwrap().is_nan() {
                res_ani.push(ani);
                res_lambda.push(lambda);
            }
        }
    }
    res_ani.sort_by(|x, y| x.partial_cmp(y).unwrap());
    res_lambda.sort_by(|x, y| x.partial_cmp(y).unwrap());
    if res_ani.len() < 50 {
        return (None, None, None, None);
    }
    let suc = res_ani.len();
    let low_ani = res_ani[suc * 5 / 100 - 1];
    let high_ani = res_ani[suc * 95 / 100 - 1];
    let low_lambda = res_lambda[suc * 5 / 100 - 1];
    let high_lambda = res_lambda[suc * 95 / 100 - 1];

    return (low_ani, high_ani, low_lambda, high_lambda);
}

pub fn get_kmer_identity(seq_sketch: &SequencesSketch, estimate_unknown: bool) -> Option<f64> {
    if !estimate_unknown {
        return None;
    }

    let mut median = 0;
    let mut mov_avg_median = 0.;
    let mut n = 1.;
    for count in seq_sketch.kmer_counts.values() {
        if *count > 1 {
            if *count > median {
                median += 1;
            } else {
                median -= 1;
            }
            mov_avg_median += median as f64;
            n += 1.;
        }
    }

    mov_avg_median /= n;
    // log::debug!("Estimated continuous median k-mer count for {} is {:.3}", &seq_sketch.file_name, mov_avg_median);

    let mut num_1s = 0;
    let mut num_not1s = 0;
    for count in seq_sketch.kmer_counts.values() {
        if *count == 1 {
            num_1s += 1;
        } else {
            num_not1s += *count;
        }
    }
    //0.1 so no div by 0 error
    let eps = num_not1s as f64 / (num_not1s as f64 + num_1s as f64 + 0.1);
    //dbg!("Automatic id est, 1-to-2 ratio, 2-to-3", eps.powf(1./31.), num_1s as f64 / num_2s as f64, two_to_three);

    if mov_avg_median < MED_KMER_FOR_ID_EST && seq_sketch.mean_read_length < 400. {
        // log::info!("{} short-read sample has high diversity compared to sequencing depth (approx. avg depth < 3). Using 99.5% as read accuracy estimate instead of automatic detection for --estimate-unknown.", &seq_sketch.file_name);
        return Some(0.995f64.powf(seq_sketch.k as f64));
    }

    if eps < 1. {
        return Some(eps);
    } else {
        return Some(1.);
    }
}
