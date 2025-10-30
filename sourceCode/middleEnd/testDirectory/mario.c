#include <cs50.h>
#include <stdio.h>

// Prototypes of Helper Functions
int get_height(void);
void make_pyramid(int n);

int main(void)
{
    // Get height of the desired pyramid between 1 and 8
    int height = get_height();

    // Prints the desired pyramid
    make_pyramid(height);
}

// Returns an integer between 1 and 8
int get_height(void)
{
    int i;
    do
    {
        i = get_int("Height: ");
    }
    while (i < 1 || i > 8);
    return i;
}

// Prints the desired pyramid of height n
void make_pyramid(int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n + (i + 3); j++)
        {
            if (j < n - (i + 1) || j > n + (i + 2) || j == n || j == n + 1)
            {
                printf(" ");
            }
            else
            {
                printf("#");
            }
        }
        printf("\n");
    }
}
