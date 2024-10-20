from collections import Counter
import os
from tqdm import tqdm
import pandas as pd

from ppievo.io import read_fasta, get_protein_lengths, write_dict
from ppievo.utils import DATA_DIR
from ppievo.datasets.human_ppi import get_all_interacting_pairs


def _write_pair_msa_num_sequences(
    pair_msa_dir: str = os.path.join(DATA_DIR, "pair_msa"),
    output_path: str = os.path.join(
        DATA_DIR, "cache/metadata/pair_msa_num_sequences.json"
    ),
) -> dict:
    """
    Get the number of sequences in all pair MSAs.
    Only need to run once
    """
    all_pairs = get_all_interacting_pairs()
    num_sequences = {}
    for protein1, protein2 in tqdm(all_pairs):
        name = f"{protein1}_{protein2}"
        msa_path = os.path.join(pair_msa_dir, f"{name}.fas")
        msa = dict(read_fasta(msa_path))
        num_sequences[name] = len(msa)
    write_dict(num_sequences, output_path)
    return num_sequences


def calc_msa_stats(pair_msa_dir=os.path.join(DATA_DIR, "pair_msa")):
    msa_stats_df = []
    # Iterate over all MSA files
    all_pairs = get_all_interacting_pairs()
    for protein1, protein2 in tqdm(all_pairs):
        name = f"{protein1}_{protein2}"
        msa_path = os.path.join(pair_msa_dir, f"{name}.fas")
        msa = dict(read_fasta(msa_path))
        query_seq = msa[f"{protein1}\t{protein2}"]
        protein1_len, protein2_len = get_protein_lengths(protein1, protein2)
        msa_stats_df.append(
            {
                "name": name,
                "protein1": protein1,
                "protein2": protein2,
                "num_sequences": len(msa),
                "length": len(query_seq),
                "protein1_length": protein1_len,
                "protein2_length": protein2_len,
                "query_seq": query_seq,
            }
        )
    msa_stats_df = pd.DataFrame(msa_stats_df)
    return msa_stats_df


def calc_species_stats(
    pair_msa_dir: str = os.path.join(DATA_DIR, "pair_msa"),
    output_path: str = os.path.join(
        DATA_DIR, "cache/metadata/pair_msa_species_stats.csv"
    ),
) -> pd.DataFrame:

    def _calc_species_stats_msa(pair_name):
        msa_path = os.path.join(pair_msa_dir, f"{pair_name}.fas")
        msa = dict(read_fasta(msa_path))
        taxon_labels = list(msa.keys())
        total_sequences = len(taxon_labels)

        # Initialize counters
        phylum_counter = Counter()
        class_counter = Counter()
        order_counter = Counter()
        family_counter = Counter()

        primates = 0
        mammals = 0
        vertebrates = 0
        birds = 0
        insects = 0
        fungi = 0
        genomic_seqs = 0
        transcriptomic_seqs = 0

        for label in taxon_labels:
            parts = label.split()
            if parts[0].startswith("GCA"):
                genomic_seqs += 1
            elif parts[0].startswith("SRX"):
                transcriptomic_seqs += 1

            taxa = parts[1].split(":")
            if len(taxa) == 5:
                genus, family, order, class_, phylum = taxa

                phylum_counter[phylum] += 1
                class_counter[class_] += 1
                order_counter[order] += 1
                family_counter[family] += 1

                if order == "Primates":
                    primates += 1
                if class_ == "Mammalia":
                    mammals += 1
                if phylum == "Chordata":
                    vertebrates += 1
                if class_ == "Aves":
                    birds += 1
                if class_ == "Insecta":
                    insects += 1
                if phylum in ["Ascomycota", "Basidiomycota"]:
                    fungi += 1

        stats = {
            "pair_name": pair_name,
            "total_sequences": total_sequences,
            "unique_phyla": len(phylum_counter),
            "unique_classes": len(class_counter),
            "unique_orders": len(order_counter),
            "unique_families": len(family_counter),
            "primates": primates,
            "mammals": mammals,
            "vertebrates": vertebrates,
            "birds": birds,
            "insects": insects,
            "fungi": fungi,
            "genomic_sequences": genomic_seqs,
            "transcriptomic_sequences": transcriptomic_seqs,
            "primate_proportion": (
                primates / total_sequences if total_sequences > 0 else 0
            ),
            "mammal_proportion": (
                mammals / total_sequences if total_sequences > 0 else 0
            ),
            "vertebrate_proportion": (
                vertebrates / total_sequences if total_sequences > 0 else 0
            ),
        }
        return stats

    species_stats_df = []
    all_pairs = get_all_interacting_pairs()
    for protein1, protein2 in tqdm(all_pairs):
        species_stats_df.append(
            _calc_species_stats_msa(pair_name=f"{protein1}_{protein2}")
        )
    species_stats_df = pd.DataFrame(species_stats_df)
    if output_path is not None:
        species_stats_df.to_csv(output_path, index=False)
    return species_stats_df
