# CSE204-Project

Garcinuño Feliciano Angela <br>
Huima Klaara <br>
Massot Lucas 

### Playlist-based song recommender ###

input a playlist, output a playlist of other songs based on similarities with the ones inputed.

our algorithm relies on using features associated to each songs to compute distances between all of them.<br>
Once all the distances are computed we can call the function with a playlist of N songs specifying that we expect<br>
a playlist of K recommended songs. From there we take all the distances from one song of the input to all of <br>
the dataset without the inputed songs. We get a matrix with as rows all the songs in the dataset without our input,<br>
and as columns all the songs in the input, with as elements of the matrix itself the distances between the row<br>
songs and the column songs. To get the recommended playlist we take the K smallest sums over each row.<br>

the algorithm relies on 6 'features' associated with each songs :
- hottness (a number from 0-1)
- tempo (a number from 0-1)
- loudness (a number from 0-1)
- similar artists (list of similar artist based on the song)
- artist terms (list of genre associated to the song)
- artist terms weights (list of weights from 0-1 associated to each term mentionned above)
