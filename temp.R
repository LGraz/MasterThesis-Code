print("hello world") # works
print("hi world") # works
ls # fails
a <- 1
print(a)

print("hello")
extract_path_from_obj <- function(obj, path_within_obj) {
    if (length(path_within_obj) == 0) {
        return (obj)
    } else {
        extract_path_from_obj(obj[[path_within_obj[1]]], path_within_obj[-1])
    }
}

assign_to_path <- function(current_obj, path_within_obj, assignment) {
    current_entry <- path_within_obj[1]
    print(current_entry)
    if (length(path_within_obj) == 1) {
        return (current_obj[[current_entry]] <- assignment)
    }
    path_within_obj <- path_within_obj[-1]
    current_obj[[current_entry]] <- assign_to_path(current_obj[[current_entry]], path_within_obj, assignment)
    current_obj
}



fit <- lm(speed ~ ., cars)
str(fit)
fit <- assign_to_path(fit, c("model", "dist"), 1)
str(fit)
