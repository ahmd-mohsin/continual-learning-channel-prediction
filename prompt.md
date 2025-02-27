Okay so what I am trying to do is to generate wireless environment channel (H) data from quadriga. I will provide you with some files you can use that but need to write the code from scratch that does the following:

'''
[1] Firstly, it should initialize one base station and random 20 users. The base station should have UPA (uniform planner array) with 8x8 while the users should have 2x2 reciever antennas.

[2] Secondly, I should be able to change the downtilt angle of the transmitter. There should be an argument for the downtilt angle.

[3] The netwrok should be an outdoor 3GPP scenario (any standard one would work) and the carrier frequency and badnwidth everything should be 5G standard according to 3GPP.

[4] You shoul dgenerate channel matrix perfrom mone carlo simulations and save the .mat file in output directory.

[5] Take help from the qudrica files I provided, they alter the antenna and other configuarions.
''' 

Why , I want the downtilt angle as the seperate argument is because I am training a Nearual Architecture search and I need 4 different datasets with different downtilt angles. No need to loop over, i will manually change and generate synthetic channels.