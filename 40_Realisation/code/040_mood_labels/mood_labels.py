"""
Create Mode labels for the data
just little adjustments from original code from Sultana et al.
"""

import os
import pandas as pd
import numpy as np
import importlib


variables = importlib.import_module('40_Realisation.code.variables')

# cheack if the file exist nd read it read file
if os.path.isfile(variables.getSavePath("data", "040_mood_labels\data_with_mood_labels.csv")):
    variables.time_frame = pd.read_csv(variables.getSavePath("data", "040_mood_labels\data_with_mood_labels.csv"))
    print("Loaded data_with_mood_labels.csv in mood_labels.py:")
    print(variables.time_frame)
else:
    merged_frame = pd.DataFrame()  # create dataframe

    # SOURCE Sultana
    try:
        merged_frame = variables.merged_frame
        print("Loaded merged_frame successfully - script was called from execute.py")
    except AttributeError:
        print("Loading merged_frame from csv - script was called directly")
        path = variables.getSavePath("data", "030_clean_mood_data\Mood_primary_18.csv")
        merged_frame = pd.read_csv(path)

    # %%
    # Create tempral features: hour
    t_df = pd.DataFrame()  # creat empty dataframe
    t_df['timestamp'] = pd.to_datetime(merged_frame['timestamp'],
                                       unit='s')  # create new time column in the datafram
    df_time = pd.DataFrame()  # creat datafram

    df_time['hour_of_timestamp'] = t_df['timestamp'].dt.hour  # create new column  "hour of timestamp"

    # time_frame = pd.DataFrame()
    time_frame = pd.concat([merged_frame, df_time], axis=1)  # merg the two dataframes togther

    # %%
    # Assign emotional rating to each row
    # pleasant=[]
    # arousal=[]
    # dominance=[]

    # assign PAD ratings using ANEW to 49 discrete emotional states

    emotion_cols = time_frame.iloc[:, 2:51]  # empotions columns
    no_of_emotion = emotion_cols.sum(axis=1)  # count the number of emotion columns

    emotion_weights_p = time_frame.iloc[:, 2:51]  # the weight of emotion p
    emotion_weights_a = time_frame.iloc[:, 2:51]  # the weight of emotion
    emotion_weights_d = time_frame.iloc[:, 2:51]  # the weight of emotion


    # function to apply the weights on the data
    def insertEmotionWeights(emotion, p_value, a_value, d_value):
        emotion_weights_p[emotion].replace(1.0, p_value, inplace=True)
        emotion_weights_a[emotion].replace(1.0, a_value, inplace=True)
        emotion_weights_d[emotion].replace(1.0, d_value, inplace=True)


    insertEmotionWeights('ACTIVE', 6.47, 5.62, 6.32)
    insertEmotionWeights('AFRAID', 2.25, 5.12, 2.71)
    insertEmotionWeights('ALERT', 5.38, 5.14, 6.58)
    insertEmotionWeights('AMUSED', 7.05, 4.27, 5.93)
    insertEmotionWeights('ANGRY', 2.53, 6.2, 4.11)
    insertEmotionWeights('ASHAMED', 2.52, 5.65, 4.63)
    insertEmotionWeights('ATTENTIVE', 6.43, 4.37, 6.62)
    insertEmotionWeights('BORED', 2.95, 3.65, 4.96)
    insertEmotionWeights('CALM', 6.89, 1.67, 7.44)
    insertEmotionWeights('CRAZY', 5.14, 5.9, 4.19)
    insertEmotionWeights('DETERMINED', 6, 4.68, 7.09)
    insertEmotionWeights('DISGUSTED', 2.68, 4.89, 4.24)
    insertEmotionWeights('DISTRESSED', 3.38, 6.28, 4.16)
    insertEmotionWeights('DREAMY', 7.52, 4.9, 6.28)
    insertEmotionWeights('ENERGETIC', 7.57, 6.1, 5.81)
    insertEmotionWeights('ENTHUSIASTIC', 7.55, 5.9, 6.21)
    insertEmotionWeights('EXCITED', 8.11, 6.43, 7.33)
    insertEmotionWeights('FRUSTRATED', 2.55, 5.4, 3.85)
    insertEmotionWeights('GUILTY', 3.09, 4.65, 4.5)
    insertEmotionWeights('HAPPY', 8.47, 6.05, 7.21)
    insertEmotionWeights('HIGH', 5.77, 4.1, 5.76)
    insertEmotionWeights('HOSTILE', 2.35, 5.39, 4.36)
    insertEmotionWeights('HUNGRY', 3.54, 5.6, 4.59)
    insertEmotionWeights('IN_EMOTIONAL_PAIN', 2, 6.27, 3.47)
    insertEmotionWeights('IN_LOVE', 8, 5.36, 5.92)
    insertEmotionWeights('IN_PHYSICAL_PAIN', 2, 6.27, 3.47)
    insertEmotionWeights('INSPIRED', 6.89, 5.56, 7.3)
    insertEmotionWeights('INTERESTED', 6.83, 4.45, 6.83)
    insertEmotionWeights('IRRITABLE', 2.85, 6.37, 4.23)
    insertEmotionWeights('JITTERY', 3.35, 4.89, 3.5)
    insertEmotionWeights('LONELY', 2.67, 4.37, 3.33)
    insertEmotionWeights('NERVOUS', 3.56, 5.51, 4.02)
    insertEmotionWeights('NORMAL', 6.17, 2.29, 6.39)
    insertEmotionWeights('NOSTALGIC', 6.68, 4.37, 5.05)
    insertEmotionWeights('OPTIMISTIC', 7.45, 4.19, 7)
    insertEmotionWeights('PHYSICALLY_SICK', 2.29, 4.67, 2.84)
    insertEmotionWeights('PROUD', 7, 5.55, 7.09)
    insertEmotionWeights('ROMANTIC', 7.61, 5.12, 6.45)
    insertEmotionWeights('SAD', 2.1, 3.49, 3.84)
    insertEmotionWeights('SCARED', 2.8, 6.1, 4.2)
    insertEmotionWeights('SERIOUS', 5.88, 4.05, 5.67)
    insertEmotionWeights('SEXY', 7.42, 6.8, 6)
    insertEmotionWeights('SLEEPY', 4.36, 3.04, 4.56)
    insertEmotionWeights('STRESSED', 1.79, 4.72, 3.85)
    insertEmotionWeights('STRONG', 6.81, 5.14, 6.54)
    insertEmotionWeights('TIRED', 4.29, 3.67, 5.06)
    insertEmotionWeights('UNTROUBLED', 6.21, 3.02, 6.14)
    insertEmotionWeights('UPSET', 2.45, 4.49, 4.3)
    insertEmotionWeights('WORRIED', 3.27, 5.81, 4.18)

    # setup to calclate the score of pleasant,arousal and dominance
    pleasant = emotion_weights_p.sum(axis=1) / no_of_emotion
    arousal = emotion_weights_a.sum(axis=1) / no_of_emotion
    dominance = emotion_weights_d.sum(axis=1) / no_of_emotion

    # %%
    # Calculate pleasant score for each row
    time_frame.insert(loc=330, column='pleasant_score', value=pleasant)

    # %%
    # Calculate arousal score for each row
    time_frame.insert(loc=331, column='arousal_score', value=arousal)

    # %%
    # Calculate dominance score for each row
    time_frame.insert(loc=332, column='dominance_score', value=dominance)
    # %%
    # calculate dominant emotion

    # subtract 5 to scale the score between -4 to 4
    p_score = time_frame['pleasant_score'] - 5
    a_score = time_frame['arousal_score'] - 5
    d_score = time_frame['dominance_score'] - 5

    # support variables to assign labels to the data
    emo_frame = np.empty([len(time_frame.index), 3])
    emo_frame[:, 0] = p_score.abs()
    emo_frame[:, 1] = a_score.abs()
    emo_frame[:, 2] = d_score.abs()

    max_v = np.argmax(emo_frame, axis=1)

    label = np.empty([len(time_frame.index)])

    # Assign labels to the dominant emotion based on value and sign of each emotional score in 3 dimension
    for i in range(len(max_v)):
        index = max_v[i]
        if index == 0:
            if p_score.iloc[i] < 0:
                label[i] = 1  # discordant
            else:
                label[i] = 2  # pleasant
        elif index == 1:
            if a_score.iloc[i] < 0:
                label[i] = 3  # dissuade
            else:
                label[i] = 4  # aroused

        elif index == 2:
            if d_score.iloc[i] < 0:
                label[i] = 5  # submissive
            else:
                label[i] = 6  # dominance

    new_series = pd.Series(label)  # convert the list to column

    time_frame.insert(loc=333, column='emotion', value=label)  # add the lable column to dataframe "timefram"

    time_frame.drop(['Unnamed: 0'], axis=1, inplace=True)  # delete the column "Unnamed: 0"

    time_frame.drop(['ACTIVE', 'AFRAID', 'ALERT', 'AMUSED', 'ANGRY', 'ASHAMED', 'ATTENTIVE', 'BORED',
                     'CALM', 'CRAZY', 'DETERMINED', 'DISGUSTED', 'DISTRESSED', 'DREAMY', 'ENERGETIC',
                     'ENTHUSIASTIC',
                     'EXCITED', 'FRUSTRATED', 'GUILTY', 'HAPPY', 'HIGH', 'HOSTILE', 'HUNGRY', 'IN_EMOTIONAL_PAIN',
                     'IN_LOVE', 'IN_PHYSICAL_PAIN', 'INSPIRED',
                     'INTERESTED', 'IRRITABLE', 'JITTERY', 'LONELY', 'NERVOUS', 'NORMAL', 'NOSTALGIC', 'OPTIMISTIC',
                     'PHYSICALLY_SICK',
                     'PROUD', 'ROMANTIC', 'SAD', 'SCARED', 'SERIOUS', 'SEXY', 'SLEEPY', 'STRESSED', 'STRONG',
                     'TIRED',
                     'UNTROUBLED', 'UPSET', 'WORRIED',
                     'discrete:time_of_day:between0and6', 'discrete:time_of_day:between3and9',
                     'discrete:time_of_day:between6and12', 'discrete:time_of_day:between9and15',
                     'discrete:time_of_day:between12and18', 'discrete:time_of_day:between15and21',
                     'discrete:time_of_day:between18and24', 'discrete:time_of_day:between21and3'], axis=1)

    # save the file
    time_frame.to_csv(
        variables.getSavePath("data", "040_mood_labels\data_with_mood_labels.csv"))  # saee the resulte


