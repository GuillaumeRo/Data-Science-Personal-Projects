################### Loading Libraries #########################################
import http.client
import json
import os



#Set working directory
os.chdir("D:\\Users\\station\\Documents\\GitHub\\Data-Science-Personal-Projects\\Predicting football matches outcomes\\API Requests")

################# API Request #################################################

connection = http.client.HTTPConnection('api.football-data.org')
headers = { 'X-Auth-Token': '61de0e7c6d6e4d95a18dad217acbdee9', 'X-Response-Control': 'minified' }
connection.request('GET', '/v1/competitions', None, headers )
response = json.loads(connection.getresponse().read().decode())

print (response)

################# Convert JSON to table #######################################

import json2table
from json2table import convert
json_object = {"key" : "value"}
build_direction = "LEFT_TO_RIGHT"
table_attributes = {"style" : "width:100%"}
html = convert(json_object, build_direction=build_direction, table_attributes=table_attributes)
print(html)


################# Saving response in a file ###################################

# Open a file for writing
out_file = open("test.json","w")

#Saving JSON file
json.dump(response, out_file)

out_file.close()



############### Opening a JSON file ###########################################

# Open the file for reading
in_file = open("test.json","r")

# Load the contents from the file, which creates a new dictionary
new_dict = json.load(in_file)

# Close the file... we don't need it anymore  
in_file.close()