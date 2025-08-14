"""
### Sentspace 0.0.2 (C) 2020-2022 [EvLab](evlab.mit.edu), MIT BCS. All rights reserved.

Homepage: https://sentspace.github.io/sentspace

For questions, email:

`{gretatu,asathe} @ mit.edu`

.. include:: ../README.md
"""

from pathlib import Path

import sentspace.utils as utils
import sentspace.syntax as syntax
import sentspace.lexical as lexical
# import sentspace.embedding as embedding

# from sentspace.Sentence import Sentence

import pandas as pd
from functools import reduce
from itertools import chain
from tqdm import tqdm


def run_sentence_features_pipeline(
    input_file: str,
    stop_words_file: str = None,
    benchmark_file: str = None,
    output_dir: str = None,
    output_format: str = None,
    batch_size: int = 2_000,
    process_lexical: bool = False,
    process_syntax: bool = False,
    process_embedding: bool = False,
    process_semantic: bool = False,
    parallelize: bool = True,
    # preserve_metadata: bool = True,
    syntax_server: str = "http://localhost/",
    syntax_port: int = 8000,
    limit: float = float("inf"),
    offset: int = 0,
    emb_data_dir: str = None,
) -> Path:
    """
    Runs the full sentence features pipeline on the given input according to
    requested submodules (currently supported: `lexical`, `syntax`, `embedding`,
    indicated by boolean flags).

    Returns an instance of `Path` pointing to the output directory resulting from this
    run of the full pipeline. The output directory contains Pickled or TSVed pandas
    DataFrames containing the requested features.


    Args:
        input_file (str): path to input text file containing sentences
                            one per line [required]
        stop_words_file (str): path to text file containing stopwords to filter
                                out, one per line [optional]
        benchmark_file (str): path to a file containing a benchmark corpus to
                                compare the current input against; e.g. UD [optional]

        {lexical,syntax,embedding,semantic,...} (bool): compute submodule features? [False]
    """

    # lock = multiprocessing.Manager().Lock()

    # create output folder
    utils.io.log("creating output folder")
    output_dir = utils.io.create_output_paths(
        input_file, output_dir=output_dir, stop_words_file=stop_words_file
    )
    # config_out = output_dir / "this_session_log.txt"
    # with config_out.open('a+') as f:
    #     print(args, file=f)

    utils.io.log("reading input sentences")
    sentences = utils.io.read_sentences(input_file, stop_words_file=stop_words_file)
    utils.io.log("---done--- reading input sentences")

    for part, sentence_batch in enumerate(
        tqdm(
            utils.io.get_batches(
                sentences, batch_size=batch_size, limit=limit, offset=offset
            ),
            desc="processing batches",
            total=len(sentences) // batch_size + 1,
        )
    ):
        sentence_features_filestem = f"sentence-features_part{part:0>4}"
        token_features_filestem = f"token-features_part{part:0>4}"

        ################################################################################
        #### LEXICAL FEATURES ##########################################################
        ################################################################################
        if process_lexical:
            utils.io.log("*** running lexical submodule pipeline")
            _ = lexical.utils.load_databases(features="all")

            lexical_features = utils.parallelize(
                lexical.get_features,
                sentence_batch,
                wrap_tqdm=True,
                desc="Lexical pipeline",
                max_workers=None if parallelize else 1,
            )

            lexical_out = output_dir / "lexical"
            lexical_out.mkdir(parents=True, exist_ok=True)
            utils.io.log(f"outputting lexical token dataframe to {lexical_out}")

            # lexical is a special case since it returns dicts per token (rather than per sentence)
            # so we want to flatten it so that pandas creates a sensible dataframe from it.
            token_df = pd.DataFrame(chain.from_iterable(lexical_features))

            if output_format == "tsv":
                token_df.to_csv(
                    lexical_out / f"{token_features_filestem}.tsv", sep="\t", index=True
                )
                token_df.groupby("sentence").mean().to_csv(
                    lexical_out / f"{sentence_features_filestem}.tsv",
                    sep="\t",
                    index=True,
                )
            elif output_format == "pkl":
                token_df.to_pickle(
                    lexical_out / f"{token_features_filestem}.pkl.gz", protocol=5
                )
                token_df.groupby("sentence").mean().to_pickle(
                    lexical_out / f"{sentence_features_filestem}.pkl.gz", protocol=5
                )
            else:
                raise ValueError(f"output format {output_format} not known")

            utils.io.log("--- finished lexical pipeline")

        ################################################################################
        #### SYNTAX FEATURES ###########################################################
        ################################################################################
        if process_syntax:
            utils.io.log("*** running syntax submodule pipeline")

            syntax_features = [
                syntax.get_features(
                    sentence,
                    dlt=True,
                    left_corner=True,
                    syntax_server=syntax_server,
                    syntax_port=syntax_port,
                )
                for i, sentence in enumerate(
                    tqdm(sentence_batch, desc="Syntax pipeline")
                )
            ]

            # put all features in the sentence df except the token-level ones
            token_syntax_features = {"dlt", "leftcorner"}
            sentence_df = pd.DataFrame(
                [
                    {
                        k: v
                        for k, v in feature_dict.items()
                        if k not in token_syntax_features
                    }
                    for feature_dict in syntax_features
                ],
                index=[s.uid for s in sentence_batch],
            )

            # output gives us dataframes corresponding to each token-level feature. we need to combine these
            # into a single dataframe
            # we use functools.reduce to apply the pd.concat function to all the dataframes and join dataframes
            # that contain different features for the same tokens
            token_dfs = [
                reduce(
                    lambda x, y: pd.concat([x, y], axis=1, sort=False),
                    (v for k, v in feature_dict.items() if k in token_syntax_features),
                )
                for feature_dict in syntax_features
            ]

            for i, df in enumerate(token_dfs):
                token_dfs[i]["index"] = df.index
            #     token_dfs[i].reset_index(inplace=True)

            dicts = [
                {k: v[list(v.keys())[0]] for k, v in df.to_dict().items()}
                for df in token_dfs
            ]
            token_df = pd.DataFrame(dicts)
            token_df.index = token_df["index"]
            # by this point we have merged dataframes with tokens along a column (rather than just a sentence)
            # now we need to stack them on top of each other to have all tokens across all sentences in a single dataframe
            # token_df = reduce(lambda x, y: pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)]), token_dfs)
            # token_df = token_df.loc[:, ~token_df.columns.duplicated()]

            syntax_out = output_dir / "syntax"
            syntax_out.mkdir(parents=True, exist_ok=True)
            utils.io.log(f"outputting syntax dataframes to {syntax_out}")

            if output_format == "tsv":
                sentence_df.to_csv(
                    syntax_out / f"{sentence_features_filestem}.tsv",
                    sep="\t",
                    index=True,
                )
                token_df.to_csv(
                    syntax_out / f"{token_features_filestem}.tsv", sep="\t", index=True
                )
            elif output_format == "pkl":
                sentence_df.to_pickle(
                    syntax_out / f"{sentence_features_filestem}.pkl.gz", protocol=5
                )
                token_df.to_pickle(
                    syntax_out / f"{token_features_filestem}.pkl.gz", protocol=5
                )
            else:
                raise ValueError(f"unknown output format {output_format}")

            utils.io.log("--- finished syntax pipeline")

        # Calculate PMI
        # utils.GrabNGrams(sent_rows,pmi_paths)
        # utils.pPMI(sent_rows, pmi_paths)

        # Plot input data to benchmark data
        # utils.plot_usr_input_against_benchmark_dist_plots(df_benchmark, sent_embed)

    ################################################################################
    #### \end{run_sentence_features_pipeline} ######################################
    ################################################################################
    return output_dir
