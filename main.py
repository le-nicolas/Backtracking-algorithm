def is_safe(board, row, col, n):
    # Check this column on the upper side
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on the left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on the right side
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_nqueens_util(board, row, n):
    # If all queens are placed, return True
    if row >= n:
        return True

    # Try placing this queen in all columns one by one
    for col in range(n):
        if is_safe(board, row, col, n):
            # Place this queen in board[row][col]
            board[row][col] = 1

            # Recur to place the rest of the queens
            if solve_nqueens_util(board, row + 1, n):
                return True

            # If placing queen in board[row][col] doesn't lead to a solution, then remove the queen (backtrack)
            board[row][col] = 0

    # If the queen cannot be placed in any column in this row, return False
    return False

def solve_nqueens(n):
    # Initialize the board
    board = [[0 for _ in range(n)] for _ in range(n)]

    if not solve_nqueens_util(board, 0, n):
        print("Solution does not exist")
        return False

    # Print the solution
    for row in board:
        print(" ".join("Q" if col == 1 else "." for col in row))
    return True

# Example usage:
n = 4
solve_nqueens(n)


#future work, i want to visualize the process. kind of like a graphviz sort of thing
