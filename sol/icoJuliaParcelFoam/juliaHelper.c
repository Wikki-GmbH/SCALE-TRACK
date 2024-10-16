#include <julia.h>
#include <stdio.h>

void jl_eval_string_with_exception(const char* str)
{
    JL_TRY {
        const char filename[] = "none";
        jl_value_t *ast = jl_parse_all(str, strlen(str),
                filename, strlen(filename), 0);
        JL_GC_PUSH1(&ast);
        jl_toplevel_eval_in(jl_main_module, ast);
        JL_GC_POP();
        jl_exception_clear();
    }
    JL_CATCH {
        jl_value_t *errs = jl_stderr_obj();
        printf("A Julia exception was caught\n");
        if (errs) {
            jl_value_t *showf = jl_get_function(jl_base_module, "showerror");
            if (showf != NULL) {
                jl_call2(showf, errs, jl_current_exception());
                jl_printf(jl_stderr_stream(), "\n");
            }
        }
        exit(1);
    }
}

/* Helper function to retrieve pointers to cfunctions on the Julia side. */
void *get_cfunction_pointer(const char *name)
{
    void *p = 0;
    jl_value_t *boxed_pointer = jl_get_global(jl_main_module, jl_symbol(name));

    if (boxed_pointer != 0)
    {
        p = jl_unbox_voidpointer(boxed_pointer);
    }

    if (!p)
    {
        fprintf(stderr, "cfunction pointer %s not available.\n", name);
    }

    return p;
}
