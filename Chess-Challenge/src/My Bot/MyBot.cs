using ChessChallenge.API;
using System;
using System.Linq;
using System.IO;

public class MyBot : IChessBot
{
    Move[] TT = new Move[8388608];
    Move bestRootMove;

    static int[] getWeights()
    {
        byte[] fileBytes = File.ReadAllBytes("./src/My Bot/simple-10.bin");
        int[] temp = new int [197377];
        for (int i = 0; i < fileBytes.Length; i += 2)
        {
            byte hiByte = fileBytes[i];
            byte lowByte = fileBytes[i + 1];
            short val = (short)((hiByte << 8) | lowByte);
            if (i/2 <= 197376) temp[i / 2] = val;
        }
        return temp;
    }
    static int[] weights = getWeights();
    public static int nInput = 768; 
    public static int nHidden = 256;
    public static double[,] iNodes = new double[2, 768];
    public static double [] hNodes = new double[nHidden];
    public static double oNode;
    public double[,] ihWeights = assignihWeights();
    public double[,] hoWeights = assignhoWeights();
    public double[] hBiases = assignhBiases();
    public static double oBias = weights [(nInput + 3) * nHidden];
    
    static double[,] assignihWeights()
    {
        int[] tempweights = getWeights();
        double[,] temp = new double[nInput, nHidden];
        for (int i = 0; i < nInput; i++)
        for (int j = 0; j < nHidden; j++) 
        {
            temp[i, j] = tempweights[i * nHidden + j];
        }
        return temp;
    }
    static double[,] assignhoWeights()
    {
        double[,] temp = new double[2, nHidden];
        for (int i = 0; i < nHidden; i++)
        {
            temp[0, i] = weights[nInput * nHidden + nHidden + i];
            temp[1, i] = weights[nInput * nHidden + nHidden * 2 + i];
        }
        return temp;
    }
    static double[] assignhBiases()
    {
        double[] temp = new double[nHidden];
        for (int i = 0; i < nHidden; i++)
        {
            temp[i] = weights[nInput * nHidden + i];
        }
        return temp;
    }
   

    public Move Think(Board board, Timer timer)
    {
        var killers = new Move[128];
        var history = new int[6, 64];
        double search(int depth, double alpha, double beta, int ply)
        {
            var (moveIndex, extension, reduction, key, inQSearch, bestMove, score, pieces) = (0, 0, 0, board.ZobristKey % 8388608, depth <= 0, Move.NullMove, 0d, board.AllPiecesBitboard);
            if (board.IsInCheck()) extension++;
            score = evaluate();
            if (inQSearch) alpha = Math.Max(alpha, (int)score);
            if (depth < 6 && score - 31.8 * Math.Max(depth, 0) >= beta) return score; //rfp and in qsearch stand pat to reduce tokens
            foreach (Move move in board.GetLegalMoves(inQSearch).OrderByDescending(move => 
                        (move == TT[key],
                         move.IsCapture ? (long)move.CapturePieceType * 10_000_000_000_000_000 - (long)move.MovePieceType :
                         move == killers[ply] ? 5_000_000_000_000_000 :
                         history[(int)move.MovePieceType - 1, move.TargetSquare.Index])))
            {
                if (inQSearch && score < alpha - (move.IsPromotion ? 1800 : 1000)) return alpha; //Does Deltapruning really gain
                board.MakeMove(move);
                double value = board.IsDraw() ? 0
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
        double evaluate()
        {
            for (int piecetype = 1; piecetype <= 6; piecetype++)
            {
                for (int c = -1; c <= 1; c += 2)
                {
                    ulong piecesBitboard = board.GetPieceBitboard((PieceType)piecetype, c == 1);
                    while (piecesBitboard != 0)
                    {
                        int index = BitboardHelper.ClearAndGetIndexOfLSB(ref piecesBitboard);
                        if (c == 1)
                        {
                            iNodes[board.IsWhiteToMove ? 0 : 1, 64 * piecetype - 1 + index] = 1;
                            iNodes[board.IsWhiteToMove ? 1 : 0, 64 * piecetype + 5 + index] = 1;
                        }
                        else
                        {
                            iNodes[board.IsWhiteToMove ? 0 : 1, 64 * piecetype + 5 + index] = 1;
                            iNodes[board.IsWhiteToMove ? 1 : 0, 64 * piecetype - 1 + index] = 1;
                        }                        
                    }
                }
            }

            double evaluation = oBias; // Add the bias

            for (int x = 0; x < 2; x++)
            {
                for (int j = 0; j < nHidden; j++)
                    hNodes[j] = 0.0;
                oNode = 0.0;

                for (int j = 0; j < nHidden; j++)
                {
                    for (int i = 0; i < nInput; i++)
                        hNodes[j] += ihWeights[i, j] * iNodes[x, i];
                    hNodes[j] += hBiases[j];  // Add the bias
                    hNodes[j] = Math.Clamp(hNodes[j], 0, 255);  // Activation
                }

                for (int j = 0; j < nHidden; j++)
                    oNode += hoWeights[x, j] * hNodes[j];
                
                evaluation += oNode;
            }
            evaluation *= 400;
            evaluation /= 255*64;
            return evaluation;
        }
        int i = 0;
        while (timer.MillisecondsElapsedThisTurn < timer.MillisecondsRemaining/26) search(++i, -2000000001, 2000000001, 0);
        return bestRootMove;
    }
}