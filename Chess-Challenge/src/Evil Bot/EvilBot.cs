using ChessChallenge.API;
using System;
using System.Linq;
using static System.Convert;
using static System.Math;

namespace ChessChallenge.Example
{
    public class EvilBot : IChessBot
    {
    Move[] TT = new Move[8388608];
    Move bestRootMove;
    int[] endgametransitionpiecevalue = { 0, 1, 1, 2, 4, 0};
    int[][] table = new int[][]{ //mg + eg, with material value baked in
    //Pawn
    new int[] {6160466, 6160466, 6160466, 6160466, 6160466, 6160466, 6160466, 6160466, 17825972, 17498328, 16515215, 14942385, 15794326, 14811344, 16973940, 18415687, 12320844, 12714073, 11731052, 10551409, 9830547, 9633930, 11534443, 11665470, 8257604, 7733343, 7012440, 6488167, 6029417, 6422622, 7274595, 7274555, 7012407, 6750288, 5963853, 5701726, 5701731, 5636184, 6357084, 6094905, 6422584, 6619214, 5767246, 6225992, 6160469, 5832789, 6094963, 5636166, 7012399, 6684753, 6684734, 6815803, 7012419, 6160490, 6291576, 5701692, 6160466, 6160466, 6160466, 6160466, 6160466, 6160466, 6160466, 6160466,},
    //Knight
    new int[] {14614698, 15925496, 17563951, 16580896, 16384398, 16646384, 14287170, 11927782, 16777480, 17891624, 16777625, 18284917, 17826152, 16777615, 16843096, 15008064, 16843042, 17105293, 19071350, 19005842, 18350501, 17826258, 17170842, 15729021, 17301832, 18612578, 19857764, 19857798, 19857782, 19136918, 18940259, 17236327, 17236292, 18022741, 19464545, 20054366, 19464557, 19530084, 18678118, 17236297, 16908602, 18219336, 18350429, 19399003, 19071332, 18219362, 17105258, 16974145, 15663412, 17105180, 17760581, 18088270, 18284880, 17105251, 16908611, 15532350, 16515304, 15073596, 16908567, 17432880, 16974144, 17236277, 15139134, 14221626,},
    //Bishop
    new int[] {18547024, 18088305, 18743579, 18940232, 19005780, 18874691, 18350452, 17891685, 18940243, 19202429, 19923291, 18678112, 19267979, 18612648, 19202431, 18547006, 19595613, 18940306, 19464600, 19399061, 19333520, 19857823, 19464594, 19726699, 19267945, 20054386, 20251008, 20054431, 20382098, 20119954, 19661172, 19595627, 19071335, 19661178, 20316538, 20709767, 19923343, 20119929, 19267959, 18874737, 18678125, 19267964, 19988860, 20119932, 20316539, 19661192, 19005823, 18481527, 18547057, 18284924, 19005821, 19399021, 19726708, 18874754, 18481550, 17695086, 17957196, 18874730, 17957215, 19136856, 18874720, 18415969, 19136838, 18350424,},
    //Rook
    new int[] {34406909, 34210311, 34734589, 34538000, 34341404, 34341350, 34079228, 33882632, 34275832, 34406909, 34406935, 34275867, 33358381, 33751584, 34079223, 33751561, 34013656, 34013680, 34013687, 33882625, 33817070, 33358346, 33227290, 33358317, 33817029, 33751506, 34406884, 33620471, 33686005, 33620480, 33489365, 33685961, 33751481, 33882563, 34079185, 33817052, 33227238, 33161686, 33030627, 32833990, 33292720, 33554884, 33227213, 33489356, 33096160, 32768477, 33030616, 32506300, 33161649, 33161677, 33554889, 33685972, 32965084, 32965096, 32834007, 33358230, 32965066, 33685968, 33751518, 33489390, 33227245, 32702948, 33817016, 32244163,},
    //Queen
    new int[] {60752869, 62784513, 62784542, 63112205, 63112252, 62587949, 61998124, 62653486, 60228585, 62653402, 63439868, 64029698, 65143793, 62981178, 63308829, 61342775, 60031988, 61735920, 61932552, 64553993, 64422942, 63636537, 62587952, 61932602, 61539302, 62784486, 62915569, 64291825, 65078272, 63964178, 65078271, 63702018, 60163064, 63177703, 62587896, 64422903, 63374335, 63570941, 63898628, 62850046, 60294131, 59573251, 62325750, 61735935, 61932540, 62456835, 61998095, 61670406, 59900894, 59835385, 59376652, 60294147, 60294153, 59835408, 58983422, 59245570, 59180032, 59507695, 59900920, 58524683, 61015026, 59245544, 60031970, 58655695,},
    //King
    new int[] {-4849729, -2293737, -1179632, -1179663, -720952, 983006, 262146, -1114099, -786403, 1114111, 917484, 1114105, 1114104, 2490364, 1507290, 720867, 655351, 1114136, 1507330, 983024, 1310700, 2949126, 2883606, 851946, -524305, 1441772, 1572852, 1769445, 1703906, 2162663, 1703922, 196572, -1179697, -262145, 1376229, 1572825, 1769426, 1507284, 589791, -720947, -1245198, -196622, 720874, 1376210, 1507284, 1048546, 458737, -589851, -1769471, -720889, 262136, 851904, 917461, 262128, -327671, -1114104, -3473423, -2228188, -1376244, -720950, -1835000, -917532, -1572840, -262130,},
    };

    public Move Think(Board board, Timer timer)
    {
        var killers = new Move[128];
        var history = new int[6, 64];
        int search(int depth, int alpha, int beta, int ply)
        {
            var (moveIndex, extension, reduction, key, inQSearch, bestMove, score, pieces, evalValues) = (0, 0, 0, board.ZobristKey % 8388608, depth <= 0, Move.NullMove, 0, board.AllPiecesBitboard, new [] {0ul, 943240312410411277ul, 4197714699149851955ul, 4848484849616963136ul, 6658122805863343458ul, 17289018720893200097ul, 508351539015584769ul, 2313471533096915729ul, 4777002364955480891ul, 5717702758025484112ul, 9909758167411563417ul, 17073413321325017080ul, 1447370843669012753ul,});
            if (board.IsInCheck()) extension++;
            score = evaluate();
            if (inQSearch) alpha = Math.Max(alpha, score);
            if (depth < 6 && score - 31.8 * Math.Max(depth, 0) >= beta) return score; //rfp and in qsearch stand pat to reduce tokens
            foreach (Move move in board.GetLegalMoves(inQSearch).OrderByDescending(move => 
                        (move == TT[key],
                         move.IsCapture ? (long)move.CapturePieceType * 10_000_000_000_000_000 - (long)move.MovePieceType :
                         move == killers[ply] ? 5_000_000_000_000_000 :
                         history[(int)move.MovePieceType - 1, move.TargetSquare.Index])))
            {
                if (inQSearch && score < alpha - (move.IsPromotion ? 1800 : 1000)) return alpha; //Does Deltapruning really gain
                board.MakeMove(move);
                int value = board.IsDraw() ? 0
                        :   board.IsInCheckmate() ? 20000 - ply
                        :   -search(depth - 1 + extension - reduction, -beta, -alpha, ply + 1);
                board.UndoMove(move);
                if (timer.MillisecondsElapsedThisTurn > timer.MillisecondsRemaining / 13) return 42;
                if (value > alpha)
                {
                    alpha = value;
                    bestMove = move;
                }
                if (alpha >= beta)
                {
                    if (!move.IsCapture)
                    {
                        killers[ply] = move;
                        history[(int)move.MovePieceType - 1, move.TargetSquare.Index] += depth * depth;
                    }
                    break;
                } 
            }
            if (ply == 0) bestRootMove = bestMove;
            TT[key] = bestMove;
            return alpha;
        }
        int evaluate()
        {
            int result = 0,
                gamePhase = 0;

            for (int piecetype = 1; piecetype <= 6; piecetype++)
            {
                for (int c = -1; c <= 1; c += 2)
                {
                    ulong piecesBitboard = board.GetPieceBitboard((PieceType)piecetype, c == 1);
                    while (piecesBitboard != 0)
                    {
                        int index = BitboardHelper.ClearAndGetIndexOfLSB(ref piecesBitboard);
                        result += table[piecetype - 1][index ^ 28 + 28 * c] * c;
                        gamePhase += endgametransitionpiecevalue[piecetype - 1];
                    }
                }
            }
            if (!board.IsWhiteToMove)
            {
                result = -result;
            }
            int mgPhase = gamePhase;
            if (mgPhase > 24) mgPhase = 24; /* in case of early promotion */
            int egPhase = 24 - mgPhase;
            return ((short)result * mgPhase + (short)((result + 0x8000) >> 16) * egPhase) / 24;
        }
        int i = 0;
        while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining/26) search(++i, -20001, 20001, 0);
        return bestRootMove;
    }
}}