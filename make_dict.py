import pandas as pd

'mean_bandwidth': ([3138.20695765]),
 'mean_centroid': ([3296.77333343]),
 'mean_flatness': ([0.05204113]),
 'mean_ioi': ([0.6640907]),
 'mean_mfcc': ([-291.71202245,   40.16217534,   41.10678984,   25.4730596 ,
          8.4089992 ,   12.32404358,    1.75544578,    9.45137945,
         11.39893421,   17.10759748,   13.89246443,   15.44219256,
          9.32216839]),
 'mean_onset': ([1.95870965]),
 'mean_rolloff': ([7175.97105875]),
 'mean_zcr': ([0.15200788]),
 'std_bandwidth': ([355.05479358]),
 'std_centroid': ([1487.39287899]),
 'std_flatness': ([0.07518581]),
 'std_ioi': ([0.01137541]),
 'std_mfcc': ([84.46219929, 34.2960946 , 11.56306835, 12.50023466, 13.38570309,
       13.44962418,  8.85633439,  6.80111643, 11.30705599,  9.29555882,
        8.6563477 ,  9.52431752,  8.26442198]),
 'std_onset': ([2.96298111]),
 'std_rolloff': ([1879.41810347]),
 'std_zcr': ([0.14933016]),
 'tempo': ([89.10290948])

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(x)

# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)