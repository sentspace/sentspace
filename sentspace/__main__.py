#!/usr/bin/env python

import argparse
import json
import pathlib
import sys
from distutils.util import strtobool

import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sentspace
import sentspace.utils as utils
from itertools import chain
from functools import reduce, partial


def main(**kwargs):
    """used to run the main pipeline, start to end, depending on the arguments and flags"""
    # Parse input
    parser = argparse.ArgumentParser(
        "sentspace",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="path to input file or a single sentence. If "
        "supplying a file, it must be .csv .txt or .xlsx,"
        " e.g., example/example.csv",
    )

    # Add an option for a user to include their own stop words
    parser.add_argument(
        "-sw",
        "--stop_words",
        default=None,
        type=str,
        help="path to delimited file of words to filter out from analysis, e.g., example/stopwords.txt",
    )

    # Add an option for a user to choose their benchmark
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="sentspace/benchmarks/lexical/UD_corpora_lex_features_sents_all.csv",
        help="path to csv file of benchmark corpora For example benchmarks/lexical/UD_corpora_lex_features_sents_all.csv",
    )

    # parser.add_argument('--cache_dir', default='.cache', type=str,
    #                  	help='path to directory where results may be cached')

    parser.add_argument(
        "-p",
        "--parallelize",
        default=True,
        type=strtobool,
        help="use multiple threads to compute features? "
        "disable using `-p False` in case issues arise.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="./out",
        type=str,
        help="path to output directory where results may be stored",
    )

    parser.add_argument(
        "-of", "--output_format", default="pkl", type=str, choices=["pkl", "tsv"]
    )

    # parser.add_argument('-id', '--request_id', type=str, default=None, required=False,
    #                     help='Provide a custom identifier string to mark unique requests made to the program')
    # Add an option for a user to choose to not do some analyses. Default is true
    parser.add_argument(
        "-lex",
        "--lexical",
        type=strtobool,
        default=False,
        help="compute lexical features? [False]",
    )
    parser.add_argument(
        "-con", "-syn",
        "--contextual", "--syntax",
        type=strtobool,
        default=False,
        help="compute syntactic features? [False]",
    )
    # parser.add_argument(
    #     "-emb",
    #     "--embedding",
    #     type=strtobool,
    #     default=False,
    #     help="compute high-dimensional sentence representations? [False]",
    # )
    # parser.add_argument(
    #     "-sem",
    #     "--semantic",
    #     type=strtobool,
    #     default=False,
    #     help="compute semantic (multi-word) features? [False]",
    # )

    # parser.add_argument(
    #     "--emb_data_dir",
    #     default="/om/data/public/glove/",
    #     type=str,
    #     help="path to output directory where results may be stored",
    # )
    # parser.add_argument('--cache_dir', default=)


    parser.add_argument(
        "--syntax_server",
        type=str,
        default="http://localhost",
        help="The URL where the Syntax module's server is running. "
        "The syntax module requires a parser server based on "
        "https://github.com/sentspace/sentspace-syntax-server.",
    )

    parser.add_argument(
        "--syntax_port",
        type=int,
        default=8000,
        help="The port where Syntax module's parser is running. "
        "The syntax module requires a parser server based on "
        "https://github.com/sentspace/sentspace-syntax-server.",
    )

    parser.add_argument(
        "--limit",
        type=float,
        default=float("inf"),
        help="Limit input to X sentences in total",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip the first X sentences"
    )

    args = parser.parse_args()

    utils.io.log(f"SENTSPACE. Received arguments: {args}")

    # dummy call with no arguments just to get the output dir
    output_dir = sentspace.run_sentence_features_pipeline(
        args.input_file, output_dir=args.output_dir
    )
    with (output_dir / "STATUS").open("w+") as f:
        f.write("RUNNING")

    # Estimate sentence embeddings
    try:
        output_dir = sentspace.run_sentence_features_pipeline(
            args.input_file,
            stop_words_file=args.stop_words,
            benchmark_file=args.benchmark,
            process_lexical=args.lexical,
            process_syntax=args.contextual,
            # process_embedding=args.embedding,
            # process_semantic=args.semantic,
            output_dir=args.output_dir,
            output_format=args.output_format,
            parallelize=args.parallelize,
            # TODO: return_df or return_path?
            # emb_data_dir=args.emb_data_dir,
            syntax_server=args.syntax_server,
            syntax_port=args.syntax_port,
            limit=args.limit,
            offset=args.offset,
        )
    except Exception as e:
        with (output_dir / "STATUS").open("w+") as f:
            f.write("FAILED")
        raise e

    with (output_dir / "STATUS").open("w+") as f:
        f.write("SUCCESS")


if __name__ == "__main__":
    main()
