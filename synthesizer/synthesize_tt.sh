echo 'synthesizing MUStt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_music.cfg

echo 'synthesizing CAFtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_cafe.cfg

echo 'synthesizing STRtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_STR.cfg

echo 'synthesizing BUStt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_bus.cfg

echo 'synthesizing PEDtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_PED.cfg

echo 'synthesizing fowltt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_fowl.cfg

echo 'synthesizing fantt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_fan.cfg

echo 'synthesizing birdtt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_bird.cfg

echo 'synthesizing doortt'
python noisyspeech_synthesizer_multiprocessing_v2.py --cfg noisyspeech_synthesizer_tt_door.cfg

