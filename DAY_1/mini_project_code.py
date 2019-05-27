



import random

tries = 1
def game():
    while tries <= 6:
        if guess_number == computer_number:
            print("You win! Computer lose!!")            
        else:
            print("You Lose! Your number was {} and Computer Win! Secret Number is {} ".format(guess_number,computer_number))
            print("You have {} trial left.".format(tries))
            if guess_number > computer_number:
                print("too high")
            else:
                print("too low")
                break
                
computer_number = random.randint(0,10)

guess_number = int(input("enter your number"))

game()            



