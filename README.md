## Nathaniel Su (nbs63)

# Introduction and Rationale

This is a more detailed description of what I did in regards to clustering similar vocal
signals together on the Northern Congo data with the intention to find a way to retrain
BirdNET on multiple different target species so that performance, and specifically recall,
could be improved. This investigation is based on a [paper](https://www.sciencedirect.com/science/article/pii/S1574954125002791?utm#s0010) I found that retrained birdNET
based on their own acoustic monitoring grid, which utilized a clustering method to
quickly find and select training samples. I found this idea interesting and wanted to try to
implement it on my target species, or the Noudable-Ndoki grid, because prior
performance evaluation showed that the detector generally had poor performance and
recall on an individual day basis. Using feature embeddings extracted by birdNET, I
created a Python script based on the method that this paper used to see if their method
could be successfully applied to the Noudable-Ndoki grid.
My intent behind this report is to serve as a tutorial/instructions overview of what I did
and how to hopefully replicate it. It also serves as documentation for what I did in the
context of retraining birdNET for my target bird species on the Noudable-Ndoki grid in
Northern Congo. The sections concerning using birdNET to retrain are more of my
notes/observations of the process and what worked best for me through the many
iterations that I went through. I’ll include what I found were the best birdNET settings for
both retraining processes. The clustering segment of this report is more of a tutorial on
how to use the tool and how to fine-tune the clustering algorithm through changing
parameters within the python script.

# General workflow

1. Give birdNET training samples to differentiate between target and non-target
    species
2. Run batch analysis on a large dataset to extract sound clips that contain as much
    of the target species as possible
3. Extract embeddings using BirdNET using the sound clips exported from the
    batch analysis
4. Download and run the Python script for clustering
5. Setup embeddings and selection table (and sound folder) in correct file structure for clustering
6. Have a directory of all the embeddings exported from birdNET, which also
    contains a birdNET selection table with corresponding selection numbers so that
    a selection table containing the labels can also be exported (more on formatting
    later)
7. Run the clustering algorithm on the embeddings (the Python script for this has
    many parameters that can be changed, something I’ll get into later)
8. Manually review each cluster label using Raven Pro to figure out which clusters
    are target species and which ones are just noise (Try to mess with settings to
    have an optimum number of clusters. A large number of clusters is time
    consuming to go through)
9. Use these clusters to develop training samples to retrain BirdNET again on your
    target species, which hopefully will help improve classifier performance
10. Assess detector performance through validation against hand-browsed
    selections of target species

# First Retraining

The first step in this process is to retrain BirdNET to essentially distinguish between
target and non-target species. Its purpose is to essentially find more training samples
within a dataset to retrain the final model on. Although hand-browsed annotations are
used to train this “binary” birdNET model, there shouldn’t be any conflict with validating
the final retrained model with this same data because the final training samples are
going to be taken from days that were not hand-browsed. Hand browsing is solely used
for finding separate training samples that in the end should have no relation to the
original hand-browsed samples, especially since not all samples found through
clustering based on this “binary” birdNET will be used for the final retraining
In principle, this probably would only necessitate two classes: “target” and “nontarget”
(noise). However, choosing the right noise classes is very finicky and often breaks the
retrained model, especially if two very different types of signals are included in the same
noise folders. I found that splitting up the noise/non-target signals into their own classes
helped a lot to prevent the detector from annotating every possible 3-second segment
as a detection. This requires a lot of experimenting and running batch analysis on a
smaller subset of recordings as a sanity check to make sure the retrained model is
actually working properly. You could also run the clustering on all your combined noise
samples to further group similar noise signals together.
Since the result of this first retraining is used to feed into the clustering algorithm, it is
better to cast a “wider” net, as it is better to have more false positives of non-target
signals if that means more of the target signals are also caught. Since clustering will
happen anyway, it is easier to filter out non-target signals through that process.


# Clustering

## Setup and Required File Structure

**Application .exe file (reccomended)**
This is recommended because of its simplicity and also because you don’t need to
install python or its dependencies for the tool to run. Download the latest clustering app
from the releases page [here](https://github.com/Nat-Su-lemon/BirdNET_Clustering/releases/). Just simply run the .exe file from wherever you downloaded it from
and you should be all set. Just keep in mind that sometimes it takes a bit to load,
especially the first time loading it up

**Running from Python Source Code**
This method gives more insight and control into the code. First, make sure to have a
recent version (3.10+) of Python installed on your machine as well as a code editor of
your preference. Download the source code zipped folder from the same link above and
unzip those contents into a directory of your choice. To install the required
dependencies, first open up a terminal (either powershell or command prompt), and
change directory into the folder where you downloaded the source code. Then, make
sure you have a ‘requirements.txt’ file in that directory. To install the dependencies, type:
**pip install -r requirements.txt**
Then type:
**Python ClusterApp.py**
To run the tool
Running either from the application file or source code should yield both a terminal
window and the gui, which looks like this:


As you can see, most of the buttons here are self explanatory, and the purposes of the
settings themself are both explained in short tooltips that can be accessed by hovering


over the settings and also are explained more in depth later in this document. While
there are some restrictions to the types of values that can be input, there is not a whole
lot of protection against putting in faulty values that break the program. I would
recommend keeping an eye on the terminal if you are trying new settings in case they
cause the program to crash. If that happens, just exit the program and restart (the most
foolproof way to cancel any ongoing processes).
You can also save and load presets of the settings you choose to use. These settings
are stored in a .json file and you can also choose where to store them.
Below are a description of each setting

**Embedding_path:** This is where the birdNET embeddings are stored. They should be a
list of text files extracted through the birdNET embeddings feature contained in a folder.
These text files essentially contain a 1024 long list of floating point numbers that the the
algorithm uses for clustering. If the sound clips are longer than 3 seconds, then birdNET
creates multiple embeddings for that one sound. This script only takes the first 3
seconds of embeddings and discards the rest.

**Selection_name:** Set this to a raven selection table saved as a tab delimited file. In
order for this to be useful, the soundclips that the embeddings are extracted from need
to be extracted from only this selection table so that the selection numbers can match
up. In order for the selections to match up, when you extract sound clips from raven to
get their embeddings, make sure that the sound clip files start with sel.xx.... Where xx is
the corresponding selection number. Set to None if you don't want this
What do I mean by one selection table? You can extract clips from multiple sound files,
but they have to be first loaded into raven from just one single selection table before
exporting the sound clips. Excel can be used to combine multiple selection tables into
one.

**Base_folder_name:** This is what the base folder for the clustering output should be
called. The final output folder name depends on the number of duplicates and clustering
settings.

**Save_path:** The folder where all outputs are saved in. The script will create a folder
with this name if it does not exist


**Input_path:** The directory containing the original sound files that the embeddings were
extracted from. This is required if sort_sounds is set to true. If it is set to true, the script
will take the sounds from this folder and copy them into their own label clusters in the
output directory. The sound clip names should match their embeddings, which should
be a given anyway since you have to extract the clips normally from raven selection
tables.

**Recluster_noise:** Set this to true if you want to recluster the noise again. Not
necessary for the most part
Both the embedding and input path have to exist and contain sound/text files with
corresponding names. You can leave on the text added on by the birdNET embeddings
extraction as the script will deal with that itself. The most important thing about the file
names of the embeddings is that the sel.xx selection number stays the same. This is
how the script knows which embedding it is currently dealing with!

## Algorithm Parameters

The clustering algorithm I am using for this script is called HDBSCAN. It is a
density-based clustering algorithm that is able to output noise and can operate without
specifying a cluster size. I am using the Scikit-learn implementation of this algorithm
[(documentation can be found here)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

Another type of algorithm I am using is something called [UMAP](https://umap-learn.readthedocs.io/en/latest/). It serves the purpose
of reducing the dimensionality of data, since HDBSCAN doesn’t work very well with
high-dimensional data such as birdNET embeddings (which are 1024D). Typically, I
reduce the dimensions to the 10 to 50 dimension range, where it seems like the
clustering algorithm works the best. It is important to figure out how to fine-tune the
parameters to suit your particular dataset. Results can vary drastically depending on the
settings you choose, and often, you need to manually verify some of the clustering first
before settling with the preferred set of parameters
The specific parameters that can be changed can be found in the script itself in the top
section labeled “Parameter Config”. Most of these are already explained in comments,
but I will go over some of the observed effects of the most important parameters that
can be changed.


**Cluster Selection Method**
This is perhaps the most important and impactful setting that can be changed. The
default setting is ‘eom’, and it can be changed to ‘leaf’. Think about the difference
between these two as casting a wider versus a smaller net. The former tends to have
very few clusters and is able to group more similar sound signals together, while the
latter tends to create more subgroups when signals are slightly different. The ‘leaf’
method does a better job of finding rarer types of signals and clustering them together,
especially in the context of larger datasets where some signals could be more prevalent
than others, but it also puts more sounds into the noise cluster, which could have the
risk of excluding many sounds that contain the target species. The ‘eom’ method, on the
other hand, tends to group more similar sounds together and label fewer sounds as
noise, but it is susceptible to labeling wrong and different signals together. Generally, I
would say the ‘leaf’ method misses a lot of signals, while the ‘eom’ method can
incorrectly group some signals together. What works best is, unfortunately, left up to trial
and error, so some manual double-checking may be needed

**Number of Components (Dimension to reduce to)**
This is basically the end dimension that the UMAP tool will reduce the embeddings to.
Generally speaking, it is best to reduce the dimension to somewhere around 10-50, but
I have not seen that much of a relation between the results and the dimension to which
the embeddings are reduced. The most important thing to know is that it is often
necessary, especially if you are using the ‘eom’ method, to reduce the dimensionality of
your data.
Since the next few settings concern the number of clusters and cluster size, I may use
the term conservative. When I say conservative, I mean that the clustering is more
local–there are smaller clusters due to fine differences between sound signals. Less
conservative clusters are larger and consist of more generally similar signal
characteristics.
The exact definitions of these parameters can be found either in the comments or
through their specific documentation websites. What I am writing down is just my
observations of the effects that changing the parameters will have on clustering of the
datasets I’ve been working with (Noudable-Ndoki ELP grid).

**Min Samples/Min Cluster Size**
These parameters go hand in hand with selecting larger more broad clusters vs smaller
more distinct clusters. Having a higher minimum cluster size results in fewer but more
broad clusters while a smaller cluster size is able to include smaller more detailed
signals that otherwise would get excluded to noise. 

Minimum samples control how dense a cluster has to be to be considered a cluster. I noticed that increasing the min
sample size tends to result in increasing the amount of noise. Hence, for my use cases,
I’ve found that lowering the min sample and cluster sizes works the best for picking up
more subtle differences and call types. If your goal is to have a more specie wide
clustering of dominant and stereo typed vocal signals, then having higher min
samples/cluster size values may prove to be more beneficial

**Cluster Selection Epsilon**
This setting also seems to impact the amount of clusters observed. While the default is
zero, increasing this parameter value by even a tenth cuts down on the number of
clusters and will tend to groups more similar sounds together, which could be either
beneficial or not depending on your target species vocal types.

**Number of Neighbors**
This is an UMAP setting that also concerns how conservative/fine-grained the clusters
will be, with a lower number being more conservative and a higher number resulting in
larger groups that are more resistant to noise.
These are the settings I include in the parameter config section of the script. Other
settings/parameters can also be set through editing the specific method calls for
HDBSCAN and UMAP themselves. You can find more documentation about these
algorithms on their respective documentation websites and see if any additional
parameter changes could improve the clustering process. Another point to mention is
the noise reclustering that I added at the end of the code. This is enabled by a true/false
settings in the parameter configuration and it essentially does clustering again on any
sounds labeled as noise and them outputs a selection table of the clustered noise in its
own folder within the output. This only works if you included a valid single raven
selection table that includes all the embeddings


## Output

Above is an example of what the cluster output looks like. If the sort_sounds parameter
is set as true, then individual folders will be created or each label and the corresponding
sounds will be copied over from the source sounds directory. This is both time and
storage intensive so its not advisable to do this for larger datasets. It may be easier
though for smaller data or if you don’t want to extract training clips from raven pro.
Additionally, there are three excel files that output which files correspond to which labels
as well as a documentation of clustering settings

## General Observations

Through developing this clustering tool, I observed that often working with the
parameters was very finicky and needed a large amount of trial and error to find
something that suited the dataset that I was working with. The clustering algorithm,
when tuned right, is very good at grouping similar signals together, but also can very
easily throw signals into the noise category or create a new cluster group for slightly
different signals or signals that contain more than one vocal call. Often, it is imperative
to check not just the label folder containing calls of your target signals but also the
adjacent labels with close numbers, because often similar sounds can be grouped


together. For example, if a tinkerbird species is grouped into label folder 88, then there
is a good chance that label folders up to a few numbers above or below 88 also contain
tinkerbirds, albeit much less of them.
I also went into a lot of trouble working with the algorithm for larger groups of
embeddings. The clustering is affected by the amount of sounds in total, and settings
need to be changed so that groups are neither too broad that there are alot of false
positives or that they aren’t too specific where the majority of signals are thrown into the
noise cluster.
When testing which parameters work the best, it is advised to keep all but one
parameter constant so that it is easier to discern the effect that the parameters may
have on the clustering data. Working with this tool is basically just defining the tradeoff
between having to annotate too many clusters vs having to filter out false positives due
to large cluster size. Sometimes for the sake of time, it may be wise to still include a
lower cluster minimum size but just don’t go through any clusters under a certain
number (unless they contain an adjacent number to a cluster that contains target
signals)

# Second Retraining

The second retraining is the more important one, where the intent is to maximize the
recall and minimize false positives. I found that the most important part of this process
was to find good representative noise training samples as well as ensure a roughly
equal number of samples per training class. A heavily unbalanced set of classes, along
with poor selections for noise signals, can drastically reduce the effectiveness of having
good examples for your training classes
The separation of any differences between noise signals is beneficial. It is also helpful
to use trial and error to determine noise signals to include. Retraining the model and
then running batch analysis on a few hours of sounds can help identify potentially
confounding signals that need to be included as a noise class
Clustering of noise signals could also prove beneficial sometimes, so that each noise
class isn’t too dissimilar. The clustering of the embeddings could be helpful as well
because not all signals used for embeddings are necessarily the target species.


As for training parameters, I found that autotune generally provided the best results. You
can also view the parameters it chose in the retrained model output files. Often, the
better settings probably vary depending on the training samples provided so autotune is
more efficient
Some specific issues that I ran into where with the selection of noise classes. It is hard
to predict what types of noise signals may get confused with your target species. Often,
I had to resort to trial and error by first running a retrained model on a subset of a day,
but that is very time consuming. Furthermore, you have to be sure that any subset you
use is representative of the larger dataset that you plan to run the detector on. There
could be unknown signals that could make recall worse without you realizing.
