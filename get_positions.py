import sys
from tqdm import tqdm
sys.path.append('/scratch/jbm96361/sarracen')
import sarracen as sc
import os
import numpy as np
import pandas as pd
import math




#I am doing this to get the relevant phantom dumps
combined_set = [f'{num:05}' for num in list(range(0, 1000, 2))]
prefix = 'planet0' # edit this based on the Run number
current_directory = os.getcwd()
nop = 3 #number of planets gives number of rows
 #len(combined_set) gives number of column

#initialize an empty array of that shape
results= np.zeros((nop, len(combined_set))) 

#polar angles
angles= np.zeros((nop, len(combined_set))) 

y=0  


for i in tqdm(combined_set, desc="Extracting Relevant information from Phantom Dump Files Files"):
    phantom_filename = current_directory+f"/{prefix}_{i}"
    if os.path.exists(phantom_filename):
  
        try:
            sdf, sdf_sinks = sc.read_phantom(f'{prefix}_{i}') #extracts the dust and sink particle data
            #Number of planets data is given by len(sdf_sinks)-1, the first row is for the star and the other 2 are for the planets. 
            # I need to calculate the position of the planet in each dumpfile by using the x,y, and z coordinates
            
            for i in range(1,len(sdf_sinks)):
               
                #I am iterating through the rows which contain data about the planets. The first row has data about the star
                row = sdf_sinks.iloc[i]
                #coordinates = row[['x', 'y', 'z']] 
                position = np.sqrt(row['x']**2 + row['y']**2 + row['z']**2 )
                results[i-1,y] = position
                angles[i-1,y] = math.degrees(math.atan(row['y']/row['x']))
            y=y+1
                

            
            
            
        except Exception as e:
            print(f"Error reading PHANTOM file {phantom_filename}: {e}")
        # Handle the error as needed
        continue  # Continue to the next iteration
        
        
    else:
        print(f"The directory {phantom_filename} does not exist.")

#create a comprhensive Dataframe to view results
# Generate row labels
row_labels = [f'planet{i+1}' for i in range(nop)]

# Generate column labels
column_labels = [f'pos{i+1}' for i in range(len(combined_set))]
# Convert the 2D array to a DataFrame with dynamic labels
df = pd.DataFrame(results, columns=column_labels, index=row_labels)

# Save DataFrame to a CSV file
df.to_csv('planet_positions.csv', index=False)


