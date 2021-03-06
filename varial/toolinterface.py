import os
import time
import inspect

import analysis
import diskio
import monitor
import settings
import wrappers
from util import ResettableType, deepish_copy


class _ToolBase(object):
    """
    Base class for post processing.
    """
    can_reuse = False
    io = diskio

    def __init__(self, tool_name=None):
        super(_ToolBase, self).__init__()

        # name
        if not tool_name:
            self.name = self.__class__.__name__
        else:
            self.name = tool_name

        # messenger
        self.message = monitor.connect_object_with_messenger(self)

    def __enter__(self):
        analysis.push_tool(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        analysis.pop_tool()

    def reset(self):
        pass

    def wanna_reuse(self, all_reused_before_me):
        """If True is returned, run is not called."""
        return self.can_reuse and all_reused_before_me

    def starting(self):
        self.message.started()

    def run(self):
        pass

    def finished(self):
        self.message.finished()

    @staticmethod
    def lookup(key, default=None):
        return analysis.lookup(key, default)


class Tool(_ToolBase):
    """Tool is the host for your business code."""
    __metaclass__ = ResettableType
    can_reuse = True

    def __init__(self, name=None):
        super(Tool, self).__init__(name)
        self.cwd = None
        self.result = None
        self.logfile = None
        self.time_start = None
        self.time_fin = None

    def __enter__(self):
        self.reset()
        res = super(Tool, self).__enter__()
        self.cwd = analysis.cwd
        self.logfile = os.path.join(self.cwd, '%s.log' % self.name)
        return res

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cwd = None
        self.logfile = None
        super(Tool, self).__exit__(exc_type, exc_val, exc_tb)

    def wanna_reuse(self, all_reused_before_me):
        return (
            super(Tool, self).wanna_reuse(all_reused_before_me)
            and os.path.exists(self.logfile)
        )  # TODO make check with hash over result, stored in self.logfile

    def reuse(self):
        self.message("INFO reusing...")
        res = self.io.get('result')
        if res:
            if hasattr(res, "RESULT_WRAPPERS"):
                self.result = list(self.io.read(f) for f in res.RESULT_WRAPPERS)
            else:
                self.result = res

    def starting(self):
        super(Tool, self).starting()
        self.time_start = time.ctime() + "\n"
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

    def finished(self):
        if isinstance(self.result, wrappers.Wrapper):
            self.result.name = self.name
            self.io.write(self.result, 'result')
        elif isinstance(self.result, list) or isinstance(self.result, tuple):
            filenames = []
            for i, wrp in enumerate(self.result):
                num_str = "_%03d" % i
                filenames.append('result' + num_str)
                self.io.write(wrp, 'result' + num_str)
            self.io.write(
                wrappers.Wrapper(
                    name=self.name,
                    RESULT_WRAPPERS=filenames
                ),
                'result'
            )
        self.time_fin = time.ctime() + "\n"
        with open(self.logfile, "w") as f:    # TODO: mv log into result.info
            f.write(self.time_start)
            f.write(self.time_fin)
        super(Tool, self).finished()


class ToolChain(_ToolBase):
    """Executes PostProcTools."""

    def __init__(self, name=None, tools=None, default_reuse=False):
        super(ToolChain, self).__init__(name)
        self._reuse = default_reuse
        self.tool_chain = []
        self.tool_names = {}
        if tools:
            self.add_tools(tools)

    def reset(self):
        for t in self.tool_chain:
            t.reset()

    def add_tools(self, tools):
        for tool in tools:
            self.add_tool(tool)

    def add_tool(self, tool):
        if not isinstance(tool, _ToolBase):
            raise RuntimeError(
                '%s is not a subclass of Tool or ToolChain' % str(tool))
        if tool.name in self.tool_names:
            raise RuntimeError(
                'A tool named "%s" is already in this chain (%s).' % (
                    tool.name, self.name))
        self.tool_names[tool.name] = tool
        self.tool_chain.append(tool)

    def run(self):
        for tool in self.tool_chain:
            with tool as t:
                if tool.wanna_reuse(self._reuse):
                    tool.reuse()
                    continue
                elif tool.can_reuse:
                    if settings.only_reload_results:
                        monitor.reset()
                        raise RuntimeError('End of reload results mode at: ', t)
                    self._reuse = False

                t._reuse = self._reuse
                t.starting()
                t.run()
                t.finished()
                self._reuse = t._reuse


class ToolChainIndie(ToolChain):
    """Same as chain, but always reuses."""

    def starting(self):
        super(ToolChainIndie, self).starting()
        self._outer_reuse = self._reuse
        self._reuse = True

    def finished(self):
        self._reuse = self._outer_reuse
        del self._outer_reuse
        super(ToolChainIndie, self).finished()


class ToolChainVanilla(ToolChain):
    """
    Makes a deep copy of analysis module, restores on exit. Tools are reset.
    """
    def __enter__(self):
        res = super(ToolChainVanilla, self).__enter__()
        old_analysis_data = {}
        for key, val in analysis.__dict__.iteritems():
            if not (
                key[:2] == "__"
                or inspect.ismodule(val)
                or callable(val)
            ):
                old_analysis_data[key] = deepish_copy(val)
            else:
                old_analysis_data[key] = val
        self._old_analysis_data = old_analysis_data
        return res

    def __exit__(self, exc_type, exc_val, exc_tb):
        analysis.__dict__.clear()
        analysis.__dict__.update(self._old_analysis_data)
        del self._old_analysis_data
        super(ToolChainVanilla, self).__exit__(exc_type, exc_val, exc_tb)

    def starting(self):
        super(ToolChainVanilla, self).starting()
        self.prepare_for_systematic()
        self.message("INFO Resetting tools.")
        self.reset()

    def finished(self):
        self.finish_with_systematic()
        super(ToolChainVanilla, self).finished()

    def prepare_for_systematic(self):
        pass

    def finish_with_systematic(self):
        pass




