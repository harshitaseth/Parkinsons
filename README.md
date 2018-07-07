# Parkinsons
Run file in following order.

python feature_extraction <feature_file>  <folder or file name> <static or dynamic> <false>
   
   - This file will extract all the required features (jitter,shimmer, pitch, unvoice framed)
   - Generate a text file conatin features of all audio
   - Input should be .wav file
   
python training_parkinson.py
    - This file trained model taking UCI extracted features.
    - Train data for training
    - Test data for validation
    
    
python testing_parkinson.py
     
     - This test is used to test wheather the new generated feature performing similar to original features(UCI extracted features)
     - New extracted features are giving similar result so this can be concluded both the features contain similar information.
      
      
     

References:
1)Erdogdu Sakar, B., Isenkul, M., Sakar, C.O., Sertbas, A., Gurgen, F., Delil, S., Apaydin, H., Kursun, O., 'Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings', IEEE Journal of Biomedical and Health Informatics, vol. 17(4), pp. 828-834, 2013.

2) T. Arias-Vergara, J. C. Vásquez-Correa, J. R. Orozco-Arroyave, Parkinson’s Disease and Aging: Analysis of Their Effect in Phonation and Articulation of Speech, Cognitive computation, (2017).

3)J. R. Orozco-Arroyave, J. C. Vásquez-Correa et al. "NeuroSpeech: An open-source software for Parkinson's speech analysis." Digital Signal Processing (2017).
