import pickle as pkl
"""
Mannually setup norm vectors for each dataset
"""

# MSACFG -- Trainset
maxVector = [2, 1367, 5411, 525, 0, 4245, 1, 2516938, 8875, 2489922, 7916, 25527, 2516954]
minVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
avgVector = [0.5962032061725268, 0.4180271544042307, 1.6158560684958867, 0.24502194444083092, 0.0, 2.175661078432198, 0.13442428699219405, 63.45916465925378, 2.262125637901996, 78.41444585456485, 0.018334971530191924, 1.9924283931335767, 70.89752509521182]
stdVector = [0.49097240095840666, 1.3898312633734076, 27.59374888091898, 0.514249527342708, 0.000000000000000, 17.252417053163388, 0.3411076048066078, 8141.555766865466, 15.87218165891597, 7755.615488089883, 2.9998780886433027, 28.02481653517624, 8141.84431147715]
norm = {'minVector': minVector, 'maxVector': maxVector, 'avgVector': avgVector, 'stdVector': stdVector}
normFile = open('norm_msacfg.pkl', 'wb')
pkl.dump(norm, normFile)
normFile.close()

# MSACFG -- Testset
maxVector = [2, 1883, 6999, 1036, 0, 4122, 1, 2514158, 5407, 2486757, 4728, 25527, 2514174]
minVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
avgVector = [0.5970183273238024, 0.414436492695235, 1.6023817480537264, 0.24687678246451827, 0.0, 2.1342394812947156, 0.1330682587555883, 69.69281705685793, 2.258427280249422, 83.6536707518127, 0.018659769558083145, 1.987237055993612, 77.07021552681057]
stdVector = [0.4908864860606825, 1.5937404761957945, 27.58260356845197, 0.6617551383910477, 0.000000000000000, 17.397127884916774, 0.33964849069835007, 8474.05518627391, 15.633733132574285, 7959.109393152846, 2.8443524115866694, 24.230326277528274, 8474.337473921465]
norm = {'minVector': minVector, 'maxVector': maxVector, 'avgVector': avgVector, 'stdVector': stdVector}
normFile = open('norm_msacfg_test.pkl', 'wb')
pkl.dump(norm, normFile)
normFile.close()
