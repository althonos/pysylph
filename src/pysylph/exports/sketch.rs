use fxhash::FxHashMap;
use fxhash::FxHashSet;
use sylph::types::Kmer;
use sylph::types::BYTE_TO_SEQ;

type Marker = u32;

#[inline]
pub fn pair_kmer_single(s1: &[u8]) -> Option<([Marker; 2], [Marker; 2])> {
    let k = std::mem::size_of::<Marker>() * 4;
    if s1.len() < 4 * k + 2 {
        return None;
    } else {
        let mut kmer_f = 0;
        let mut kmer_g = 0;
        let mut kmer_r = 0;
        let mut kmer_t = 0;
        let halfway = s1.len() / 2;
        // len(s1)/2 + (k-1)* 2 + 2 < len(s1)
        for i in 0..k {
            let nuc_1 = BYTE_TO_SEQ[s1[2 * i] as usize] as Marker;
            let nuc_2 = BYTE_TO_SEQ[s1[2 * i + halfway] as usize] as Marker;
            let nuc_3 = BYTE_TO_SEQ[s1[1 + 2 * i] as usize] as Marker;
            let nuc_4 = BYTE_TO_SEQ[s1[1 + 2 * i + halfway] as usize] as Marker;

            kmer_f <<= 2;
            kmer_f |= nuc_1;

            kmer_r <<= 2;
            kmer_r |= nuc_2;

            kmer_g <<= 2;
            kmer_g |= nuc_3;

            kmer_t <<= 2;
            kmer_t |= nuc_4;
        }
        return Some(([kmer_f, kmer_r], [kmer_g, kmer_t]));
    }
}

pub fn dup_removal_lsh_full_exact(
    kmer_counts: &mut FxHashMap<Kmer, u32>,
    kmer_to_pair_set: &mut FxHashSet<(u64, [Marker; 2])>,
    //kmer_to_pair_set: &mut ScalableCuckooFilter<(u64,[Marker;2]), FxHasher>,
    //kmer_to_pair_set: &mut GrowableBloom,
    km: &u64,
    kmer_pair: Option<([Marker; 2], [Marker; 2])>,
    num_dup_removed: &mut usize,
    no_dedup: bool,
    threshold: Option<u32>,
) {
    let c = kmer_counts.entry(*km).or_insert(0);
    let mut c_threshold = u32::MAX;
    if let Some(t) = threshold {
        c_threshold = t;
    }
    if !no_dedup && *c < c_threshold {
        if let Some(doublepairs) = kmer_pair {
            let mut ret = false;
            if kmer_to_pair_set.contains(&(*km, doublepairs.0)) {
                //Need this when using approximate data structures
                if *c > 0 {
                    ret = true;
                }
            } else {
                kmer_to_pair_set.insert((*km, doublepairs.0));
            }
            if kmer_to_pair_set.contains(&(*km, doublepairs.1)) {
                if *c > 0 {
                    ret = true;
                }
            } else {
                kmer_to_pair_set.insert((*km, doublepairs.1));
            }
            if ret {
                *num_dup_removed += 1;
                return;
            }
        }
    }
    *c += 1;
}
