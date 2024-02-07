# PICO_W_neural_network_TicTacToe
Machine Learning TicTacToe Running on Raspberry Pi PICO W

Raspberry Pi PICO W で機械学習する、まるばつゲームを作りました

親指サイズのマイコン、Raspberry Pi PICO W で機械学習（AI）の実装が出来たらすごいなぁと思い、プログラムをフルスクラッチで作りました。

機械学習として、ディープラーニングの基礎概念である、ニューラルネットワークの実装をしました。

取り上げた題材はまるばつゲームです。

対戦すればするほど徐々にコンピュータが強くなっていく様子を体感下さい。

制作にあたり、MicroPythonでは行列の演算ライブラリであるNumPyが動かなかったため、NumPyの互換ライブラリも自作しました。

画面は、PICO上でWEBサーバも稼働させ、HTMLのホームページとして動くように作りました。WEBサーバは前回の動画で制作したものを活用しています。

詳しくは私のYoutube動画を見て下さい。

最後にもう一度言います。このプログラム、Raspberry Pi PICO W で動いてるんだぜ！

I created Tic Tac Toe with Machine Learning on Raspberry Pi PICO W

I thought it would be amazing if I could implement machine learning (AI) on the thumb-sized microcontroller Raspberry Pi PICO W, so I wrote the entire program from scratch.
For the machine learning implementation, I used a neural network, which is a basic concept of deep learning.
The subject I chose was the game TicTacToe.
Please experience how the computer gradually gets stronger the more you play against it.
Since I couldn't get the NumPy matrix operation library to work in MicroPython when making this, I also wrote my own library compatible with NumPy.
I made the display run as an HTML homepage, with a web server also running on the PICO. I reused the web server I made in a previous video.
Please see my Youtube video for more details.
Let me say it one more time - this program runs on the Raspberry Pi PICO W!
