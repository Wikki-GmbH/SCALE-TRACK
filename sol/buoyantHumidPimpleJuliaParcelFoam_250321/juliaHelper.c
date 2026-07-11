#include <julia.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Evaluate Julia code, reporting any exception with a full stacktrace.
 *
 * The error is rendered on the Julia side rather than the C side: showerror
 * called through the C API writes into Julia's buffered stderr, which the
 * exit(1) below discards.  Rendering in Julia also yields the stacktrace,
 * which the C API cannot easily produce.
 *
 * The wrapping try block makes a top-level assignment in the evaluated code
 * block-local, so what is passed here has to be an expression rather than an
 * assignment.  include() evaluates in module scope regardless.
 */
void jl_eval_string_with_exception(const char* str)
{
    static const char pre[] = "try\n";
    static const char post[] =
        "\ncatch __jlerr\n"
        "  Base.println(Base.stderr, \"\\nJulia exception in embedded code:\")\n"
        "  Base.showerror(Base.stderr, __jlerr, catch_backtrace())\n"
        "  Base.println(Base.stderr)\n"
        "  Base.flush(Base.stderr)\n"
        "  rethrow()\n"
        "end\n";

    const size_t n = sizeof(pre) + strlen(str) + sizeof(post);
    char* wrapped = (char*)malloc(n);
    if (!wrapped)
    {
        fprintf(stderr, "Out of memory wrapping Julia code\n");
        exit(1);
    }
    snprintf(wrapped, n, "%s%s%s", pre, str, post);

    JL_TRY {
        const char filename[] = "embedded";
        jl_value_t *ast = jl_parse_all(wrapped, strlen(wrapped),
                filename, strlen(filename), 0);
        JL_GC_PUSH1(&ast);
        jl_toplevel_eval_in(jl_main_module, ast);
        JL_GC_POP();
        jl_exception_clear();
    }
    JL_CATCH {
        printf("A Julia exception was caught\n");
        fflush(stdout);
        /* Runs Julia's atexit hooks, which flush the stderr that the catch
         * block above wrote the message and stacktrace to. */
        jl_atexit_hook(1);
        exit(1);
    }

    free(wrapped);
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
