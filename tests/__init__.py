import os
import unittest.mock
import urllib.request

print('aaaaa')
iers_data_file = os.path.join(os.path.dirname(__file__), 'data', 'finals2000A.data.csv')

# Mock urllib.request.urlopen so that tests work offline and faster.
unittest.mock.patch.object(urllib.request, 'urlopen',
                           return_value=open(iers_data_file, 'rb'))
