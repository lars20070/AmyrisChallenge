import numpy    as np
import pandas   as pd
import seaborn  as sns
import warnings as wn
import matplotlib.pyplot as plt

# ======================================== #
#     data structure for                   #
#     high-throughput screening data       #
# ======================================== #

class HTSData:
  """ class for high-throughput screening data
  """

  def __init__(self, file):
    """ imports data and collects basic information

    Args:
        file (str): absolute path to xls sheet
    """

    # load data from xls sheet
    self.df = pd.read_excel(file)
    del self.df['counter']
    self.df.index.name = 'counter'

    # set of robots
    self.robots = self.df['robot'].unique()

    # set of plates
    self.plates = self.df['plate'].unique()

    # set of strains (excluding parent strain and process control)
    self.strains = self.df[self.df['well_type']=='Standard Well']['strain'].unique()

  def get_plate(self, plate, drop_controls=False):
    """ returns 96-well plate as 12 x 8 dataframe

    Args:
        plate (int): ID number of plate
        drop_controls (bool): Should the control wells be exported as NaN?

    Returns:
        pd.DataFrame: 12 x 8 dataframe with activity values of the 96-well plate with columns [1 .. 12] and rows [A .. H]
    """

    plate = self.df[self.df['plate']==plate]
    plate = pd.DataFrame(plate['value'].values.reshape(12, 8), columns = list("ABCDEFGH")).T
    plate = plate.rename(columns={0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12})

    if drop_controls:
      plate[12]['E'] = np.nan
      plate[12]['F'] = np.nan
      plate[12]['G'] = np.nan
      plate[12]['H'] = np.nan

    return plate


# ======================================== #
#     collection of normalisation          #
#     and statistic algorithms             #
# ======================================== #

class StatsKit:
  """ collection of statistic algorithms
  """

  def __init__(self):

    # threshold for median polish
    self.threshold = 0.1

  def percent_of_control(self, htsdata):
    """ scale values in column 'value' relative to the median of the control samples on this plate

    Args:
        htsdata (HTSData): dataframe with columns 'value', 'well_type' and 'plate'

    Returns:
        pd.Series: percent of control
    """

    df = htsdata.df

    # find median of controls on each plate
    ctrl = df[df['well_type']=='Process Control']
    ctrl_avg = ctrl.groupby(ctrl['plate']).median()

    # scale the column 'value'
    dict_tmp = dict(zip(ctrl_avg.index, ctrl_avg['value']))
    poc = df['value'] / df['plate'].map(dict_tmp) * 100

    return poc

  def z_score(self, htsdata):
    """ classic z-score normalisation

    Args:
        htsdata (HTSData): dataframe with columns 'value', 'well_type' and 'plate'

    Returns:
        pd.Series: z-score normalised values
    """

    data = htsdata.df

    # data without controls
    df = data[data['well_type'] != 'Process Control']

    # mean and standard deviation for each plate
    data_avg = df.groupby(df['plate']).mean()
    dict_avg = dict(zip(data_avg.index, data_avg['value']))
    data_std = df.groupby(df['plate']).std()
    dict_std = dict(zip(data_std.index, data_std['value']))

    # shift by mean and scale by standard deviation
    zscore = (data['value'] - data['plate'].map(dict_avg)) / data['plate'].map(dict_std)

    return zscore

  def MAD(self, x):
    """ median absolute deviation

    Args:
        x (pd.Series): input series

    Returns:
        pd.Series: median absolute deviation
    """

    mad = abs(x - x.median()).median()

    return mad

  def z_score_robust(self, htsdata):
    """ robust z-score normalisation

    Args:
        htsdata (HTSData): dataframe with columns 'value', 'well_type' and 'plate'

    Returns:
        pd.Series: robust z-score normalised values
    """

    data = htsdata.df

    # data without controls
    df = data[data['well_type'] != 'Process Control']

    # median and MAD for each plate
    data_avg = df.groupby(df['plate']).median()
    dict_avg = dict(zip(data_avg.index, data_avg['value']))
    data_mad = df.groupby(df['plate']).agg(lambda x: self.MAD(x))
    dict_mad = dict(zip(data_mad.index, data_mad['value']))

    # shift by median and scale by MAD
    zscore = (data['value'] - data['plate'].map(dict_avg)) / data['plate'].map(dict_mad)

    return zscore

  def median_polish_plate(self, p):
    """ median polish a single 96-well plate

    Args:
        p (pd.DataFrame): 12 x 8 dataframe of 96-well plate

    Returns:
        pd.DataFrame: median polished 96-well plate
    """
    # median of columns and rows
    m = p.median(axis=1, skipna=True)
    n = p.median(axis=0, skipna=True)

    # repeat until row and column medians are zero
    while abs(m.abs().sum()) > self.threshold and abs(n.abs().sum()) > self.threshold:

      # subtract row medians
      m = p.median(axis=1, skipna=True)
      p = p.sub(m, axis=0)

      # subtract column medians
      n = p.median(axis=0, skipna=True)
      p = p.sub(n, axis=1)

    return p

  def b_score(self, htsdata):
    """ b-score normalisation
    96-well plate normalisation for HTS data. See Box 1 in N. Malo et al.
    Advantages: (1) non-parametric (2) removes positional effects (3) resistant to statistical outliers
    Note do not make use of controls! For example, no prior percent_of_control().

    N. Malo et al. (2006). Statistical practice in high-throughput screening data analysis. Nature Biotechnology, 24(2), 167–175. http://doi.org/10.1038/nbt1186

    Args:
        htsdata (HTSData): HTS dataset

    Returns:
        pd.Series: b-score normalised values
    """

    bscore = pd.Series([], dtype='float64')

    # loop over plates
    for p in htsdata.plates:

      # get (flat) values of plate, remove controls and calculate MAD
      plate_flat = htsdata.df[htsdata.df['plate']==p]['value']
      plate_flat = plate_flat.reset_index(drop=True).drop(index=[92, 93, 94, 95])
      mad = self.MAD(plate_flat)

      # median polish single plate
      plate = htsdata.get_plate(p, drop_controls=True)
      plate = self.median_polish_plate(plate)

      # scale by MAD
      plate = plate / mad

      # reshape polished plate
      for c in range(1,13):
        bscore = bscore.append(plate[c].squeeze())

    bscore = bscore.reset_index(drop=True)

    return bscore

  def parent_strain_correction(self, htsdata):
    """ correction relative to parent strain
    We are interested in outliers relative to the parent strain. We therefore shift the entire plate
    by the median of the parents and scale by MAD of the parents. Similar to z_score_robust(),
    but only with respect to the parent strain subset.

    Args:
        htsdata (HTSData): dataframe with columns 'value', 'well_type' and 'plate'

    Returns:
        pd.Series: parent strain corrected values
    """

    data = htsdata.df

    # parent strain wells
    parents = data[data['well_type'] != 'Parent Strain']

    # median and MAD of parents for each plate
    parents_avg = parents.groupby(parents['plate']).median()
    dict_avg    = dict(zip(parents_avg.index, parents_avg['value']))
    parents_mad = parents.groupby(parents['plate']).agg(lambda x: self.MAD(x))
    dict_mad    = dict(zip(parents_mad.index, parents_mad['value']))

    # shift by median and scale by MAD
    corrected = (data['value'] - data['plate'].map(dict_avg)) / data['plate'].map(dict_mad)

    return corrected

# ======================================== #
#     visualization of high-throughput     #
#     screening data and their analyses    #
# ======================================== #

class Visualizer:
  """ universal plotting class
  stores default parameters
  """

  def __init__(self):
    """ initializes default parameters
    """

    self.figsize = (12., 7.)
    self.palette_discrete   = 'Paired'
    self.palette_continious = 'icefire'

  def plot(self, data):
    """ plots overview

    Args:
        data (pandas.DataFrame): HTS dataset with column 'column'
    """

    plt.figure(figsize=self.figsize)
    ax = sns.scatterplot(data=data, x='counter', y='value', hue='plate', style='well_type', alpha=.5, palette=self.palette_continious, size=10)
    ax.set(xlabel='counter', ylabel='activity')
    plt.show()

  def swarmplot(self, data):
    """ plots overview

    Args:
        data (pandas.DataFrame): dataframe with index datetime and column `metric`
    """

    # ignore warning for dense swarm plot
    wn.simplefilter('ignore', category=UserWarning)

    plt.figure(figsize=self.figsize)
    ax = sns.swarmplot(data=data, x='plate', y='value', palette=self.palette_discrete, size=3)
    ax.set(xlabel='plate', ylabel='activity')
    plt.show()

  def plot_plate(self, data):
    """ plots 96-well plate

    Args:
        data (pd.Dataframe): 12 x 8 dataframe
        pdf_file (str): name of output pdf file
    """

    plt.figure(figsize=self.figsize)
    ax = sns.heatmap(data=data, cmap=self.palette_continious)
    plt.show()




