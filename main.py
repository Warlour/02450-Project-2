from ucimlrepo import fetch_ucirepo
from functions import colorize_json
import numpy as np
  
# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 

### data ###
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
### metadata ###
#print(colorize_json(rice_cammeo_and_osmancik.metadata))
  
### Variable information ###
# print(rice_cammeo_and_osmancik.variables) 

X = rice_cammeo_and_osmancik.data.features # Attribute values (features)
y = rice_cammeo_and_osmancik.data.targets # Class values (targets)

attributeNames = list(rice_cammeo_and_osmancik.data.headers)
attributeNames.remove('Class')

classNames = np.unique(y)

N = len(y)
M = len(attributeNames)
C = len(classNames)

# print(N, M, C, sep='\n')