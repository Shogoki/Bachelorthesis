import owncloud
import sys
if len(sys.argv) < 2:
	print("Please specify password")
	quit()
pw = sys.argv[1]
#print("pw is " + pw)
oc = owncloud.Client('https://skcloud.evolution-web.de')

oc.login('azure-pl', pw)
#
# oc.mkdir('testdir')

oc.put_file('output/thesis_latest.pdf', 'output/thesis.pdf')
