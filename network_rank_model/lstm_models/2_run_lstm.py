'''
8b. LSTM, CLEAN DATA: REPORTED CASES + WASTEWATER -> TRUE INFECTIONS
=====================================================================
Needs the Covasim runs made by 8a_make_covasim_runs.py. This script needs torch (not Covasim), so it can be run in a different
Python environment from 8a.

PURPOSE
  Test whether a neural network (an LSTM) can recover the TRUE daily infections of an epidemic from the two things actually observed: reported cases and wastewater concentrations. Covasim simulates the epidemics, so the true
  infections are known exactly and the estimate can be scored.
  The data are CLEAN: the reported cases are exactly what Covasim reports, and the wastewater has no unknown scale, little lab noise and a
  sample every day. The question is whether the LSTM can learn from Covasim at all, and how much each stream contributes.
  Three LSTMs are trained on the same data, each given different inputs: both streams, cases only, wastewater only.
  The comparison between them is the result.

WHAT THE LSTM SEES AND PRODUCES (for one run of 181 days)
  Input features, for every day   0  log(1 + reported cases)
                                  1  log of the wastewater concentration (gc per litre); 0 on days with no sample
                                  2  sample taken? (1 = yes, 0 = no)
                                  3  day of the week, scaled to 0-1
  The three input sets            both = features 0, 1, 2, 3   |   cases_only = 0 and 3   |   wastewater_only = 1, 2 and 3
  Output, for every day           the 5th, 50th (median) and 95th percentile of log(1 + true infections)
                                  (5th to 95th = a 90% range; log(1 + x) is used so that days with 0 infections are handled)

REPEATS
  The whole experiment is repeated N_REPEATS times. In every repeat the Covasim runs are split into train, validation and test
  in a different random way, and every LSTM starts from different random weights. The table reports the mean and the standard
  deviation over the repeats, so a difference between two settings can be judged against how much a single setting varies.
  The Covasim runs are NOT repeated: the same saved runs are used in every repeat.

LEARNING CURVE: HOW MANY RUNS ARE ENOUGH?
  TRAIN_FRACTIONS sets which shares of the training runs are used, e.g. [0.25, 0.5, 0.75, 1.0]. The validation and test runs are the
  same for every share, so the scores can be compared. If the test error is still falling at 1.0, more Covasim runs are needed
  (run 8a with more seeds; only the new seeds are run). If it has flattened, there are enough. The default [1.0] uses all of them.
  This multiplies the number of LSTMs, so for a learning curve use one stage and one input set, e.g.
  INPUT_SETS_TO_RUN = ['both'].

ASSUMPTIONS (the assumptions about how the runs were made are in 8a_make_covasim_runs.py)
  A. Reported cases
     1. A different share of cases being reported is simulated AFTERWARDS by thinning: each reported case is kept with a probability.
        The infections never change. In the clean data the probability is 1 (nothing is thrown away), so the cases are exactly Covasim's.
        The code can keep only a share of them (see keep_share in STAGES), which is not used here.
     2. There is no weekday effect and no reporting backlog in these stages.
  B. Wastewater
     3. Concentration (gc per litre) = load / (agents x 270 litres per person per day): a constant flow per person (Point Loma).
        There is no virus decay in the sewer and no detection limit.
     4. Measurement = concentration x wastewater_scale x lab noise. The wastewater is clean and reliable:
        scale 1 (the load per case is exactly right), low lab noise (Normal(mean 1, sd 0.05), cut off below at 0.05) and a sample
        every day. The noise level is a PLACEHOLDER. These settings are still in STAGES, so degraded wastewater can be tested
        later by changing them.
     5. A day with no sample or with a concentration of 0 counts as "no sample": feature 2 = 0 and feature 3 = 0.
  C. The model and its training
     6. The LSTM is bidirectional: the estimate for each day uses the WHOLE series, past and future. This is a retrospective
        estimate, like EpiNow2 and EpiSewer. It is NOT a real-time nowcast (that needs an LSTM that only sees past days).
     7. For every repeat, stage and input set a new LSTM of the same design is trained from scratch. The loss is the quantile
        (pinball) loss. The LSTMs of one stage and repeat are trained and tested on the same observed sequences, so only their
        inputs differ.
     8. The runs are split into train (70%), validation (15%) and test (15%). All sequences made from one run stay on the same
        side, so the model never sees any part of a test epidemic. The validation runs only choose the best training epoch.
        The test runs are used only for the final scores.
     9. The two features that need it are standardised with the TRAINING sequences only.
    10. All the models are trained and tested on the same observed sequences, so they differ only in the inputs they are given.
  D. Scoring
    11. All errors are on the log(1 + infections) scale.
        lstm_error_log     = mean absolute error of the median, over all days.
        coverage_90        = share of days on which the truth lies inside the 5th to 95th range. About 0.90 if the ranges are honest.
        interval_width_log = how wide that range is: narrower is better, if the coverage stays near 0.90.
    12. The baseline is crude: the wastewater signal alone (gaps filled by interpolation), moved back 4 days, plus one fixed
        offset fitted on the training runs. It is a low bar to clear, not a rival model such as EpiSewer.

WHAT TO LOOK FOR (the expectation, to be checked against the table)
  - All models should do well and beat the baseline. If not, check the pipeline before anything else.
  - cases_only and wastewater_only show how much each stream carries alone. both should be at least as good as either.
    With clean data both streams are reliable, so expect little difference between them. A large gap would mean one stream carries
    most of the information.
  - A difference only counts if it is clearly larger than the standard deviation over the repeats.

LIMITS: WHAT THESE RESULTS CAN AND CANNOT SHOW
  - Everything comes from Covasim, so nothing here shows that the model works on real data.
  - Not tested here: a share of reported cases that changes during an epidemic or is unknown from one epidemic to the next, a
    weekday pattern in reporting, noisy or sparse wastewater, a wrong load per case. The settings for these are in STAGES. A
    wastewater scale that is wrong in the SAME way in training
    and testing is simply learned by the model; what hurts in practice is a MISMATCH between the scale in training and in real
    data (train at 1, test at 0.5 and 2).
  - With few runs the test set is small (30 runs give about 5 test runs), and the scores will vary a lot between repeats.

HOW TO RUN
    pip install torch pandas matplotlib      (once; Covasim is NOT needed for this script)
    python 8b_train_lstm_stages.py
  First make the runs with 8a. At least 10 runs that took off are needed for the split. For a quick first test set N_REPEATS = 1,
  N_EPOCHS = 5 and INPUT_SETS_TO_RUN = ['both'].
  The number of LSTMs trained is N_REPEATS x len(TRAIN_FRACTIONS) x 1 stage x 3 input sets (9 with the defaults).

FILES WRITTEN (in OUTPUT_FOLDER)
    stage_scores.csv                 one row per repeat, share of training runs, stage and input set
    stage_scores_summary.csv         the mean and standard deviation over the repeats, one row per number of training runs, stage and input set
    example_<stage>_<inputs>_<n>runs.png   one test sequence per setting (from the first repeat): the truth against the 5th to 95th range
'''
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# =====================================================================================================================
# SETTINGS
# =====================================================================================================================
SCRIPT_FOLDER = Path(__file__).resolve().parent            # the folder this script is in
OUTPUT_FOLDER = SCRIPT_FOLDER / 'lstm_stages_output'       # everything is written here (the same folder as 8a)
RUNS_FOLDER   = OUTPUT_FOLDER / 'covasim_runs'             # the Covasim runs made by 8a. Use only runs from ONE setup

SEED = 1                   # the first random start. Repeat number r uses SEED + r

SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY = 270    # litres of sewage per person per day (Point Loma)

N_REPEATS           = 3    # how many times the whole experiment is repeated (a new split of the runs and new random weights each time)
TRAIN_FRACTIONS     = [1.0]    # the shares of the training runs to use. For a learning curve: [0.25, 0.5, 0.75, 1.0]
N_SEQUENCES_PER_RUN = 20   # how many different "what the analyst sees" versions are made from each run (more data for the LSTM)
N_EPOCHS            = 30   # training passes over the training data, for each LSTM
QUANTILES           = [0.05, 0.5, 0.95]                    # the three percentiles of the true infections (a 90% range)

# ---- The stages. Each number is a range (low, high): one value is drawn from it for every sequence. (1.0, 1.0) = fixed.
#   keep_share        the share of Covasim's reported cases that is kept (1.0 = none thrown away, 0.5 = about half), the same on every day (the clean data use 1.0)
#   wastewater_scale  the error in the assumed load per case: multiplies the wastewater concentration (1.0 = exactly right)
#   noise_sd          the spread of the lab noise (the noise has mean 1, so 0.1 means a typical error of about 10%)
#   schedules         which sampling schedules are used: 'daily', 'weekdays' or 'monday_thursday'
CLEAN = {'keep_share': (1.0, 1.0), 'wastewater_scale': (1.0, 1.0), 'noise_sd': (0.05, 0.05), 'schedules': ['daily']}

STAGES = {
    '1_clean': CLEAN,                                             # clean data: the starting point. Other stages can be added here
}
STAGES_TO_RUN = ['1_clean']

# ---- The inputs each LSTM is given: numbers of the four features (0 log cases, 1 log wastewater, 2 sample taken, 3 weekday)
INPUT_SETS = {'both': [0, 1, 2, 3], 'cases_only': [0, 3], 'wastewater_only': [1, 2, 3]}
INPUT_SETS_TO_RUN = list(INPUT_SETS)                              # for a quicker run use ['both']

# Which days of the week a sample is taken, for each schedule (0 = Monday ... 6 = Sunday)
SAMPLING_WEEKDAYS = {'daily': [0, 1, 2, 3, 4, 5, 6], 'weekdays': [0, 1, 2, 3, 4], 'monday_thursday': [0, 3]}


# =====================================================================================================================
# THE RUNS: loaded from the files made by 8a
# Every run is a dictionary with four entries:
#   true_infections  infections per day (the truth)         reported_cases  people diagnosed per day
#   load_gc_per_day  gene copies per day in the wastewater   n_agents        people in the simulation
# =====================================================================================================================
def load_run(file):
    """
    Read one saved Covasim run.

    Input:    file  the path of a run_seed_<seed>.npz file made by 8a
    Returns:  the run dictionary described above
    """
    saved = np.load(file)
    return {'true_infections': saved['true_infections'], 'reported_cases': saved['reported_cases'],
            'load_gc_per_day': saved['load_gc_per_day'], 'n_agents': int(saved['n_agents'])}      # int(): the file stores it as an array


def load_runs():
    """
    Load every saved Covasim run, without the runs where the epidemic died out (fewer than 1% of the agents were ever infected:
    too small to learn from). Stops if fewer than 10 runs are left, because the split into train, validation and test needs
    at least that many.

    Returns: a list of run dictionaries
    """
    files = sorted(RUNS_FOLDER.glob('run_seed_*.npz'))
    runs = [load_run(file) for file in files]
    runs = [run for run in runs if run['true_infections'].sum() > 0.01 * run['n_agents']]
    print(f'{len(runs)} of {len(files)} saved Covasim runs kept (the others did not take off)')
    assert len(runs) >= 10, 'At least 10 runs are needed to split into train, validation and test: run 8a with more seeds'
    return runs


def split_runs(runs):
    """
    Split the RUNS into train (70%), validation (15%) and test (15%) in a random order.

    The split is by run, not by day and not by sequence: all the sequences made from one run stay on the same side. If
    sequences of one epidemic were in both the training and the test set, the model would partly "remember" the test
    epidemics and the test scores would look better than they are.

    Input:    runs  a list of run dictionaries
    Returns:  train_runs, validation_runs, test_runs (three lists of run dictionaries)
    """
    run_order = np.random.permutation(len(runs))                  # a random order of the run numbers
    n_train, n_validation = int(0.7 * len(runs)), int(0.15 * len(runs))
    train_runs      = [runs[i] for i in run_order[:n_train]]
    validation_runs = [runs[i] for i in run_order[n_train:n_train + n_validation]]
    test_runs       = [runs[i] for i in run_order[n_train + n_validation:]]   # whatever is left, about 15%
    return train_runs, validation_runs, test_runs


# =====================================================================================================================
# THE OBSERVATION LAYER: what the analyst would see, for one stage
# The infections never change. Only what is OBSERVED changes, so one run gives many training sequences, and the
# ascertainment, scale and noise of every sequence are known to us (but not to the LSTM).
# =====================================================================================================================
def draw(low_high):
    """
    One random value, equally likely anywhere between low and high. If low and high are equal the value is fixed.

    Input:    low_high  a pair (low, high), e.g. (0.3, 1.0)
    Returns:  one number
    """
    return np.random.uniform(low_high[0], low_high[1])


def observe_cases(reported_cases, stage):
    """
    Simulate a different share of infections being reported, by THROWING AWAY some of Covasim's reported cases at random
    (thinning). Each reported case is kept with the same probability on every day of the run: 1.0 keeps all of them (the clean
    stage), 0.5 keeps about half of them.

    Inputs:
        reported_cases  people diagnosed per day, from Covasim (181 values)
        stage           the stage settings; stage['keep_share'] is the range the keep probability is drawn from
    Returns:  the cases that remain, per day
    """
    return np.random.binomial(reported_cases, draw(stage['keep_share']))      # for each day: how many of that day's cases are kept


def observe_wastewater(load_gc_per_day, n_agents, stage):
    """
    Turn the true wastewater load into what a lab would report: a concentration with an unknown scale, lab noise and
    missing days.

    Inputs:
        load_gc_per_day  true gene copies per day in the wastewater (181 values)
        n_agents         the number of people (the sewage volume is proportional to it)
        stage            the stage settings: wastewater_scale, noise_sd and schedules
    Returns:
        measurement   the measured concentration per day, in gc per litre
        sample_taken  True on days with a measurement, False on days without
    """
    days = np.arange(len(load_gc_per_day))
    concentration = load_gc_per_day / (n_agents * SEWAGE_FLOW_LITRES_PER_PERSON_PER_DAY)           # gc per litre, no noise

    # The scale is drawn on the log scale so that, for example, 0.5 and 2 are equally far from 1 (a factor of 2 either way)
    scale = np.exp(draw(np.log(stage['wastewater_scale'])))
    # One noise multiplier per day, around 1. clip(0.05) stops a multiplier from being zero or negative (the log is taken later).
    lab_noise = np.random.normal(1, draw(stage['noise_sd']), size=len(days)).clip(0.05)
    measurement = concentration * scale * lab_noise

    schedule = np.random.choice(stage['schedules'])                          # one schedule per sequence, picked at random
    # A sample is only taken on the days of the schedule, and a concentration of 0 (nothing shed yet) cannot be logged
    sample_taken = np.isin(days % 7, SAMPLING_WEEKDAYS[schedule]) & (measurement > 0)
    return measurement, sample_taken


def make_features(cases, measurement, sample_taken):
    """
    Put what the analyst sees into the table the LSTM reads: one row per day, four columns (features).

    Inputs (each has one value per day):
        cases         reported cases
        measurement   wastewater concentration (gc per litre)
        sample_taken  True on days with a wastewater sample
    Returns:  an array of shape (days, 4) with the columns
        0  log(1 + cases)               log(1 + x) so that days with 0 cases are fine
        1  log of the concentration     set to 0 on days without a sample (the next column says so)
        2  sample taken                 1 or 0
        3  day of the week              0 to 1, so the model can learn a weekday pattern
    """
    days = np.arange(len(cases))
    # np.maximum(..., 1e-12) only avoids log(0) on days without a concentration; multiplying by sample_taken then sets those days to 0
    log_wastewater = np.log(np.maximum(measurement, 1e-12)) * sample_taken
    return np.stack([np.log1p(cases), log_wastewater, sample_taken.astype(float), (days % 7) / 6], axis=-1)


def make_sequences(runs, stage):
    """
    Make the training material from a list of runs: N_SEQUENCES_PER_RUN observed versions of each run. Every version has
    a fresh random keep_share, wastewater scale, lab noise and sampling schedule, but the same true infections.

    Inputs:
        runs   a list of run dictionaries
        stage  the stage settings (how the observation is degraded)
    Returns:
        features  array (sequences, days, 4): what the LSTM sees
        target    array (sequences, days): the true log(1 + infections), what it should output
    """
    all_features, all_targets = [], []
    for run in runs:
        for _ in range(N_SEQUENCES_PER_RUN):
            cases = observe_cases(run['reported_cases'], stage)
            measurement, sample_taken = observe_wastewater(run['load_gc_per_day'], run['n_agents'], stage)
            all_features.append(make_features(cases, measurement, sample_taken))
            all_targets.append(np.log1p(run['true_infections']))              # the truth is the same for every version of this run
    return np.array(all_features, dtype=np.float32), np.array(all_targets, dtype=np.float32)    # float32 is what PyTorch uses


def fit_standardiser(features):
    """
    Work out the mean and standard deviation (sd) of the two features that need scaling, so they can be rescaled to mean 0
    and sd 1 (neural networks train better on inputs of about that size).

    Use the TRAINING sequences only. If the test sequences were included, information about them would leak into training.

    Input:    features  array (sequences, days, 4) of the training sequences
    Returns:  a dictionary with the mean and sd of the log cases and of the log wastewater
    """
    sampled = features[..., 2] > 0                                              # the wastewater numbers are only meaningful on days with a sample
    return {'cases_mean': features[..., 0].mean(), 'cases_sd': features[..., 0].std(),
            'wastewater_mean': features[..., 1][sampled].mean(), 'wastewater_sd': features[..., 1][sampled].std()}


def standardise(features, stats):
    """
    Rescale the log cases and the log wastewater to mean 0 and sd 1, using the numbers from fit_standardiser. The sample-taken
    flag and the weekday are left as they are. Days without a sample stay exactly 0 (the model reads the flag, not the 0).

    Inputs:   features (sequences, days, 4); stats from fit_standardiser (always from the TRAINING sequences)
    Returns:  a rescaled copy of the features, the same shape
    """
    inputs = features.copy()                                                    # copy: the original features are used again for the baseline
    inputs[..., 0] = (features[..., 0] - stats['cases_mean']) / stats['cases_sd']
    inputs[..., 1] = np.where(features[..., 2] > 0, (features[..., 1] - stats['wastewater_mean']) / stats['wastewater_sd'], 0.0)
    return inputs


# =====================================================================================================================
# THE MODEL: an LSTM and the quantile loss
# An LSTM reads a series day by day and keeps a memory of what it has seen. That suits this problem: infections today
# show up in wastewater and in reported cases over the following days, so each day's estimate needs the days around it.
# =====================================================================================================================
class InfectionLSTM(nn.Module):
    """
    Reads the features of all the days of a run and outputs, for every day, three percentiles of the log(1 + true infections).

    Input:   (sequences, days, n_features)   the features
    Output:  (sequences, days, 3)            the 5th, 50th and 95th percentile (see QUANTILES)
    """
    def __init__(self, n_features=4, hidden_size=64, n_layers=2, n_quantiles=len(QUANTILES)):
        super().__init__()
        # batch_first=True: the data are arranged (sequences, days, features). Two stacked layers, 64 memory units each.
        # dropout=0.1 randomly switches off 10% of the units between the layers during training, to reduce over-fitting.
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True,
                            bidirectional=True, dropout=0.1)           # bidirectional: each day sees the whole series
        self.to_quantiles = nn.Linear(2 * hidden_size, n_quantiles)    # 2 x because both directions are joined

    def forward(self, inputs):                                         # inputs: (sequences, days, features)
        hidden_states, _ = self.lstm(inputs)                           # (sequences, days, 2 x hidden_size)
        return self.to_quantiles(hidden_states)                        # (sequences, days, quantiles)


def pinball_loss(predicted, target):                                   # predicted: (sequences, days, quantiles), target: (sequences, days)
    """
    The quantile (pinball) loss: how wrong the three percentiles are, averaged over all days and sequences. Lower is better.

    For a percentile q, an estimate that is too LOW is penalised by q for every unit it is too low, and one that is too HIGH
    by (1 - q). Minimising this makes the 5th percentile output sit below the truth on 5% of the days, the median on 50%, and
    the 95th on 95%. That is what makes the output a calibrated range and not a single best guess.

    Inputs:   predicted (sequences, days, quantiles); target (sequences, days), the true log(1 + infections)
    Returns:  one number
    """
    quantiles = torch.tensor(QUANTILES).view(1, 1, -1)                 # shaped so it lines up with the last axis of predicted
    error = target.unsqueeze(-1) - predicted                           # positive when the estimate is too low
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def train(model, train_inputs, train_target, validation_inputs, validation_target,
          n_epochs=N_EPOCHS, learning_rate=1e-3, batch_size=32):
    """
    Train the LSTM (the model is changed in place). After every epoch the loss on the VALIDATION runs is recorded and the
    weights are saved; at the end the weights of the epoch with the lowest validation loss are put back. This stops the
    model from being kept after it has started to over-fit the training data.

    Inputs:
        model                              an InfectionLSTM
        train_inputs, train_target         the standardised training features and the true targets
        validation_inputs, validation_target   the same for the validation runs (used only to choose the best epoch)
        n_epochs       how many passes over the training data
        learning_rate  how big each update step is
        batch_size     how many sequences are used for one update
    """
    loader = DataLoader(TensorDataset(torch.tensor(train_inputs), torch.tensor(train_target)), batch_size=batch_size, shuffle=True)
    validation_inputs, validation_target = torch.tensor(validation_inputs), torch.tensor(validation_target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_losses, saved_states = [], []

    for epoch in range(n_epochs):
        model.train()                                                  # training mode: dropout is on
        for inputs, target in loader:                                  # one batch of sequences at a time, in a random order
            optimizer.zero_grad()                                      # forget the previous batch's gradients
            loss = pinball_loss(model(inputs), target)
            loss.backward()                                            # work out how each weight should change
            optimizer.step()                                           # change the weights a little

        model.eval()                                                   # evaluation mode: dropout is off
        with torch.no_grad():                                          # no gradients needed, just the loss
            validation_losses.append(pinball_loss(model(validation_inputs), validation_target).item())
        saved_states.append(copy.deepcopy(model.state_dict()))         # a copy of this epoch's weights
        print(f'  epoch {epoch + 1}: validation loss {validation_losses[-1]:.4f}')

    model.load_state_dict(saved_states[int(np.argmin(validation_losses))])    # go back to the best epoch


# =====================================================================================================================
# EVALUATION on runs the model never saw
# =====================================================================================================================
def predict_quantiles(model, inputs):
    """
    Use a trained model: for every day of every sequence, the three percentiles of log(1 + true infections).

    Inputs:   model  a trained InfectionLSTM;  inputs  standardised features (sequences, days, n_features)
    Returns:  array (sequences, days, 3): [..., 0] the 5th percentile, [..., 1] the median, [..., 2] the 95th.
              They are sorted, because the three outputs are independent and could otherwise (rarely) cross.
    """
    model.eval()
    with torch.no_grad():
        return torch.sort(model(torch.tensor(inputs)), dim=-1).values.numpy()


def wastewater_baseline(features, lag_days=4):
    """
    A deliberately simple baseline: the wastewater signal alone, with the gaps filled in, moved back by 4 days (virus shows
    up in wastewater a few days after the infection). A fixed offset is added elsewhere (see score_stage). It does not use
    the reported cases at all.

    Input:    features  array (sequences, days, 4), NOT standardised (it needs the log wastewater and the sample flag)
    Returns:  array (sequences, days), a baseline estimate of log(1 + infections) up to the offset
    """
    log_wastewater, sampled = features[..., 1], features[..., 2] > 0
    days = np.arange(features.shape[1])
    # On days without a sample, draw a straight line between the nearest samples before and after
    filled = np.array([np.interp(days, days[s], w[s]) for w, s in zip(log_wastewater, sampled)])
    # Move the signal back by lag_days (the estimate for day t is the wastewater of day t + lag_days); the last days repeat the final value
    return np.concatenate([filled[:, lag_days:], np.repeat(filled[:, -1:], lag_days, axis=1)], axis=1)


def score_stage(repeat, n_train_runs, stage_name, input_name, predicted, target, train_features, train_target, test_features):
    """
    Score one model on the test runs. All errors are on the log(1 + infections) scale.

    Inputs:
        repeat                    the number of the repeat (0, 1, 2, ...)
        n_train_runs              how many Covasim runs the model was trained on
        stage_name                the name of the stage (for the table)
        input_name                which input set the model was given: both, cases_only or wastewater_only
        predicted                 the model's three percentiles for the test sequences (sequences, days, 3)
        target                    the true log(1 + infections) of the test sequences
        train_features, train_target   used only to fit the baseline's one offset
        test_features             the test features before standardising (for the baseline)
    Returns a dictionary, one row of the table:
        coverage_90         share of days on which the truth is inside the 5th to 95th range (should be about 0.90)
        lstm_error_log      mean absolute error of the median
        baseline_error_log  the same error for the wastewater-only baseline
        interval_width_log  the average width of the range (narrower is better if the coverage is still about 0.90)
    """
    lower, median, upper = predicted[..., 0], predicted[..., 1], predicted[..., 2]
    # The baseline is on a different scale from the infections, so one number is added to line them up. It is fitted on the TRAINING runs.
    offset = (train_target - wastewater_baseline(train_features)).mean()
    return {'repeat': repeat,
            'n_train_runs': n_train_runs,
            'stage': stage_name,
            'inputs': input_name,
            'coverage_90': ((target >= lower) & (target <= upper)).mean(),                # should be about 0.90
            'lstm_error_log': np.abs(median - target).mean(),
            'baseline_error_log': np.abs(wastewater_baseline(test_features) + offset - target).mean(),
            'interval_width_log': (upper - lower).mean()}                                 # how wide the range is


def plot_example(predicted, target, title_text, path):
    """
    Save a plot of the FIRST test sequence: the true infections (black), the LSTM's median (blue) and its 5th to 95th
    percentile range (light blue), back in infections per day (expm1 undoes the log(1 + x)).

    Inputs:   predicted (sequences, days, 3); target (sequences, days); title_text for the title; path of the PNG file
    """
    days = np.arange(target.shape[1])
    plt.figure(figsize=(9, 5))
    plt.fill_between(days, np.expm1(predicted[0, :, 0]), np.expm1(predicted[0, :, 2]), color='#AEC8DE', label='5th to 95th percentile')
    plt.plot(days, np.expm1(predicted[0, :, 1]), color='steelblue', linewidth=2, label='LSTM median')
    plt.plot(days, np.expm1(target[0]), color='black', linewidth=2, label='Covasim truth')
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Infections per day', fontsize=16)
    plt.title(f'LSTM estimate vs true infections ({title_text})', fontsize=18)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =====================================================================================================================
# ONE STAGE, THEN ALL OF THEM, THEN THE REPEATS
# =====================================================================================================================
def run_stage(repeat, stage_name, train_runs, validation_runs, test_runs):
    """
    Everything for one stage in one repeat: make the observed sequences (once, so every input set sees exactly the same data),
    standardise, then for each input set train a NEW LSTM from scratch, predict on the test runs and score.

    Inputs:   repeat (0, 1, 2, ...); stage_name (a key of STAGES); the three lists of runs from split_runs for this repeat
              (train_runs may be only a share of the training runs, see TRAIN_FRACTIONS)
    Returns:  a list of score rows, one per input set (dictionaries, see score_stage)
    """
    print(f'\n=== repeat {repeat + 1} of {N_REPEATS}, {stage_name}: {STAGES[stage_name]}')
    stage = STAGES[stage_name]
    np.random.seed(SEED + repeat)                                              # a different random start in every repeat

    train_features,      train_target      = make_sequences(train_runs, stage)
    validation_features, validation_target = make_sequences(validation_runs, stage)
    test_features,       test_target       = make_sequences(test_runs, stage)

    stats = fit_standardiser(train_features)                                   # from the training sequences only
    train_inputs      = standardise(train_features, stats)
    validation_inputs = standardise(validation_features, stats)
    test_inputs       = standardise(test_features, stats)

    scores = []
    for input_name in INPUT_SETS_TO_RUN:
        columns = INPUT_SETS[input_name]                                       # which of the four features this model is given
        print(f'--- repeat {repeat + 1}, {stage_name}, inputs: {input_name} (features {columns})')
        torch.manual_seed(SEED + repeat)                                       # the same random weights at the start of every input set

        model = InfectionLSTM(n_features=len(columns))                         # a new, untrained model every time
        train(model, train_inputs[..., columns], train_target, validation_inputs[..., columns], validation_target)

        predicted = predict_quantiles(model, test_inputs[..., columns])
        if repeat == 0:                                                        # one example plot per setting is enough
            plot_example(predicted, test_target, f'{stage_name}, {input_name}, {len(train_runs)} runs',
                         OUTPUT_FOLDER / f'example_{stage_name}_{input_name}_{len(train_runs)}runs.png')
        scores.append(score_stage(repeat, len(train_runs), stage_name, input_name, predicted, test_target,
                                  train_features, train_target, test_features))
    return scores


def summarise(stage_scores):
    """
    The mean and the standard deviation over the repeats, for each number of training runs, stage and input set.

    Input:    stage_scores  the table with one row per repeat, stage and input set
    Returns:  a table with one row per number of training runs, stage and input set, and columns such as lstm_error_log_mean and lstm_error_log_std
              (the standard deviation is empty if there was only one repeat)
    """
    metrics = ['coverage_90', 'lstm_error_log', 'baseline_error_log', 'interval_width_log']
    summary = stage_scores.groupby(['n_train_runs', 'stage', 'inputs'], sort=False)[metrics].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{statistic}' for metric, statistic in summary.columns]
    return summary.reset_index()


def main():
    """Load the runs once, then repeat: split the runs, and for each share of the training runs run every stage on that split.
    Save all the scores and their summary."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    runs = load_runs()

    all_scores = []
    for repeat in range(N_REPEATS):
        np.random.seed(SEED + repeat)
        train_runs, validation_runs, test_runs = split_runs(runs)              # a different split in every repeat
        print(f'\nrepeat {repeat + 1}: runs: train {len(train_runs)} | validation {len(validation_runs)} | test {len(test_runs)}')
        for fraction in TRAIN_FRACTIONS:
            used_train_runs = train_runs[:int(fraction * len(train_runs))]     # the runs are already in a random order
            assert len(used_train_runs) >= 2, f'TRAIN_FRACTIONS has {fraction}, which leaves fewer than 2 training runs'
            for stage_name in STAGES_TO_RUN:
                all_scores += run_stage(repeat, stage_name, used_train_runs, validation_runs, test_runs)

    stage_scores = pd.DataFrame(all_scores)
    stage_scores.to_csv(OUTPUT_FOLDER / 'stage_scores.csv', index=False)
    summary = summarise(stage_scores)
    summary.to_csv(OUTPUT_FOLDER / 'stage_scores_summary.csv', index=False)

    print(f'\nSummary over {N_REPEATS} repeat(s) (errors are on the log scale; coverage_90 should be about 0.90):')
    print(summary.round(3).to_string(index=False))
    print(f'\nSaved in {OUTPUT_FOLDER}')


if __name__ == '__main__':
    main()
