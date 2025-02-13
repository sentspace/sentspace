# sentspace
<img src="graphics/logo_v1.png" width="300">

<!-- ABOUT THE PROJECT -->
## About 

`sentspace` is an open-source tool for characterizing text using diverse features related to how humans process and understand language. 
`sentspace` characterizes textual input using cognitively motivated lexical, syntactic, and semantic features computed at the token- and sentence level. Features are derived from psycholinguistic experiments, large-scale corpora, and theoretically motivated models of language processing.
The `sentspace` features fall into two core modules: Lexical and Contextual. The Lexical module operates on individual lexical items (words) within a sentence and computes a summary representation by combining information across the words in the sentence. This module includes features such as frequency, concreteness, age of acquisition, lexical decision latency, contextual diversity, etc.
The Contextual module operates on sentences as whole and includes syntactic features, such as the depth of center embedding. Note that using the contextual module requires additional set up steps (see in the setup section below). 

New modules can be easily added to SentSpace to provide additional ways to characterize text.
In this manner, `sentspace` provides a quantitative and interpretable representation of any sentence. 


**GitHub repository:** [http://github.com/sentspace/sentspace](http://github.com/sentspace/sentspace)

**Screencast video demo:** [https://youtu.be/a66_nvcCakw](https://youtu.be/a66_nvcCakw)

**CLI usage demo:**
<!-- ![image](https://i.imgur.com/lI6Wose.gif) -->
<img src="https://i.imgur.com/lI6Wose.gif" alt="drawing" width="800"/>


## [Documentation](https://sentspace.github.io/sentspace) 
<!-- [![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/main.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/main) -->
<!-- request read access to the [project doc](https://docs.google.com/document/d/1O1M7T5Ji6KKRvDfI7KQXe_LJ7l9O6_OZA7TEaVP4f8E/edit#). -->

Documentation is available online (click on the title above).


## Usage

### 1. CLI
Example: get lexical and embedding features for stimuli from a csv containing columns for 'sentence' and 'index'.
```bash
$ python3 -m sentspace -h
usage: 
                                            

positional arguments:
  input_file            path to input file or a single sentence. If supplying a file, it must be .csv .txt or .xlsx, e.g., example/example.csv

optional arguments:
  -h, --help            show this help message and exit
  -sw STOP_WORDS, --stop_words STOP_WORDS
                        path to delimited file of words to filter out from analysis, e.g., example/stopwords.txt
  -b BENCHMARK, --benchmark BENCHMARK
                        path to csv file of benchmark corpora For example benchmarks/lexical/UD_corpora_lex_features_sents_all.csv
  -p PARALLELIZE, --parallelize PARALLELIZE
                        use multiple threads to compute features? disable using `-p False` in case issues arise.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output directory where results may be stored
  -of {pkl,tsv}, --output_format {pkl,tsv}
  -lex LEXICAL, --lexical LEXICAL
                        compute lexical features? [False]
  -con CONTEXTUAL, --contextual CONTEXTUAL
                        compute syntactic features? [False]
  --emb_data_dir EMB_DATA_DIR
                        path to output directory where results may be stored
```

### 2. As a library
Example: get embedding features in a script
```python
import sentspace

s = sentspace.Sentence('The person purchased two mugs at the price of one.')
emb_features = sentspace.embedding.get_features(s)
```

Example: parallelize getting features for multiple sentences using multithreading
```python
import sentspace

sentences = [
    'Hello, how may I help you today?',
    'The person purchased three mugs at the price of five!',
    'This is an example sentence we want features of.'
             ]
             
# construct sentspace.Sentence objects from strings
sentences = [*map(sentspace.Sentence, sentences)]
# make use of parallel processing to get lexical features for the sentences
lex_features = sentspace.utils.parallelize(sentspace.lexical.get_features, sentences,
                                           wrap_tqdm=True, desc='Computing lexical features')
```


## Installing

### 1. Install using Conda and [Poetry](https://python-poetry.org/)
Prerequisites: `conda` 
1. Use your own or create new conda environment: `conda create -n sentspace-env python=3.8` (if using your own, we will assume your environment is called `sentspace-env`)
   - Activate it: `conda activate sentspace-env`
2. Install poetry: `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -` 
<!-- 3. **Manually install** (`polyglot`)[] dependencies (this step is necessary as some of the packages need to be installed using the system's package manager, rather than `conda` or `pip`) -->
3. Install `polyglot` dependencies using conda: `conda install -c conda-forge pyicu morfessor icu -y`
4. Install remaining packages using poetry: `poetry install`

If after the above steps the installation gives you trouble, you may need to refer to: [`polyglot` install instructions](https://polyglot.readthedocs.io/en/latest/Installation.html), which lists how to obtain ICU, a dependency for polyglot.

To use `sentspace` after installation, simply make sure to have the conda environment active and all packages up to date using `poetry install`
<br>

### 2. Container-based usage (Recommended!)
[![CircleCI](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci.svg?style=svg)](https://circleci.com/gh/aalok-sathe/sentspace/tree/circle-ci)

Requirements: `singularity` or `docker`. 

<!-- #### **first, some important housekeeping stuff**
- make sure you have `singularity`/`docker`, or load/install it otherwise
  - `which singularity`   or  `which docker` 
- make sure you have set the ennvironment variables that specify where `singularity/docker` will cache its images. if you don't do this, `singularity` will make assumptions and you may end up with a full disk and an unresponsive server, if running on a server with filesystem restrictions. you should have about 5GB free space at the target location. -->

<!-- #### **next, running the container** (automatically built and deployed to Docker hub) -->

**Singularity:**
```bash
singularity shell docker://aloxatel/sentspace:latest
``` 
Alternatively, from the root of the repo, `bash singularity-shell.sh`). this step can take a while when you run it for the first time as it needs to download the image from docker hub and convert it to singularity image format (`.sif`). however, each subsequent run will execute rapidly. 

**Docker:** use [corresponding commands for Docker](https://docs.docker.com/engine/reference/commandline/exec/).

now you are inside the container and ready to run `sentspace`!

### 3. Manual install (use as last resort)
On Debian/Ubuntu-like systems, follow the steps below. On other systems (RHEL, etc.), 
substitute commands and package names with appropriate alternates.
```bash
# optional (but recommended): 
# create a virtual environment using your favorite method (venv, conda, ...) 
# before any of the following

# install basic packages using apt (you likely already have these)
sudo apt update
sudo apt install python3.8 python3.8-dev python3-pip
sudo apt install python2.7 python2.7-dev 
sudo apt install build-essential git

# install ICU
DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" sudo apt install python3-icu

# install ZS package separately (pypi install fails)
python3.8 -m pip install -U pip cython
git clone https://github.com/njsmith/zs
cd zs && git checkout v0.10.0 && pip install .

# install rest of the requirements using pip
cd .. # make sure you're in the sentspace/ directory
pip install -r ./requirements.txt
polyglot download morph2.en
```



## Submodules

SentSpace features fall into two core modules: Lexical and Contextual. 
In general, each submodule implements a major class of features. 
You can run each module on its own by specifying its flag and `0` or `1` with the module call:
```bash
python -m sentspace -lex {0,1} -con {0,1} <input_file_path>
```

For a full list of available features, refer to the Feature Descriptions page [on the hosted SentSpace frontend](https://sentspace.github.io/hosted).

#### `lexical`
The Lexical module consists of features that pertain to individual lexical items, words, regardless of the context in which they appear. 
These features are returned on a word-by-word level and also aggregated at the sentence level to provide each sentence a corresponding value.

#### `contextual`
The Contextual module consists of features that quantify contextual and combinatorial inter-word 
relations that are not captured by individual lexical items. This module encompasses features that 
relate to the syntactic structure of the sentence (`Contextual_syntax` features) and features that 
apply to the sentence context but are not (exclusively) related to syntactic structure 
(`Contextual_misc` features).

**⚠ Additional steps to set up the contextual module**
The core sentspace program doesn't include a syntax server. It therefore needs to query a backend where PCFG processing can happen. 
You'll need to separately run this backend simultaneously to sentspace so that sentspace can query it and obtain features. 
The module should be running in a terminal for the duration you're using sentspace, and then you can kill it using Ctrl+C.
- Here's a link to the module: https://github.com/sentspace/sentspace-syntax-server
- Jump to the "Setup" section in the readme to run it: https://github.com/sentspace/sentspace-syntax-server?tab=readme-ov-file#setup-how-to-get-it-up-and-running
- There is a pre-built docker image so that this setup should only need 1 command (sudo docker run -it --net=host --expose 8000 -p 8000:8000 aloxatel/berkeleyparser:latest). There is also a corresponding `singularity` command for HPC cluster environs that works with the same docker image.
- This will start loading and eventually [it can take 5-10 minutes, it is slow] expose the syntax server on port 8000 (this is just a virtual address so other processes on your computer know where to look!)
- Now, sentspace can query your localhost port 8000 with sentences to be processed, and it will be returned syntax-based features for further post-processing and packaging into a nice output format similar to rest of sentspace.
- To make sure it knows to talk to the correct port, you can either pass it into the CLI (--syntax_port 8000) or as an argument to the library function: https://github.com/sentspace/sentspace/blob/4b0f79c7f6dcab6285d3af42101b04b05f421b01/sentspace/__main__.py#L127
  However, both, the library and the syntax server should default to `localhost:8000` so unless you have a special circumstance, you won't need to worry about this.


<!--
#### `embedding`
Obtain high dimensional representations of sentences using word-embedding and contextualized encoder models.
- `glove`
- Huggingface model hub (`gpt2-xl`, `bert-base-uncased`)

#### `semantic`
Multi-word features computed using partial or full sentence context.
- PMI (pointwise mutual information)
- Language model-based perplexity/surprisal
*Not Implemented yet*
-->




## Contributing

Any contributions you make are **greatly appreciated**, and no contribution is *too small* to contribute.

1. Fork the project on Github [(how to fork)](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
2. Create your feature/patch branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request (PR) and we will take a look asap!

## Whom to contact for help
- `gretatu % mit ^ edu`
- `asathe % mit ^ edu`

(C) 2020-2022 EvLab, MIT BCS
