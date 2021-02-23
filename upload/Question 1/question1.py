size = ((9, 9), (90000, 100000))
sum_list = [[65, 72, 90, 110], [87127231192, 5994891682]]
length = len(size)
with open(r"C:\Users\18834\Desktop\Question 1\output_question_1.txt", 'w', encoding='utf-8') as f:
    for i in range(length):
        m = size[i][0]
        n = size[i][1]
        max_input = m * (m + 1) / 2 + m * (n - 1)
        min_input = m * (m + 1) / 2 + n - 1
        for sum in sum_list[i]:
            if sum <= max_input and sum >= min_input:
                remainder = int((sum - m * (m + 1) / 2) % (n - 1))
                result = int((sum - m * (m + 1) / 2) // (n - 1))
                operations = ''
                if remainder != 0:
                    operations = "D" * (result - 1) + "R" * (n - 1 - remainder) + "D" * 1 + "R" * remainder + "D" * (
                            m - result - 1)
                else:
                    operations = "D" * (result - 1) + "R" * (n - 1) + "D" * (m - result)
                f.write(str(sum))
                f.write(" ")
                f.write(operations)
                f.write("\n")
            else:
                f.write("error of input\n")
        f.write("\n")