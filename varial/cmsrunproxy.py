import glob
import json
import subprocess
import time
import os
join = os.path.join

import analysis
import diskio
import monitor
import settings
import sample
import toolinterface
import wrappers


class CmsRunProcess(object):
    """
    This class hosts a cmsRun process.
    cmsRun output is streamed into logfile.
    """

    def __init__(self, sample_inst, try_reuse_data, cfg_filename):
        super(CmsRunProcess, self).__init__()

        assert isinstance(sample_inst, sample.Sample)
        if not cfg_filename:
            raise RuntimeError('CmsRunProcess needs the cfg_filename argument!')
        name = sample_inst.name
        self.sample             = sample_inst
        self.name               = name
        self.cfg_filename       = cfg_filename
        self.log_file           = None
        self.log_filename       = join(analysis.cwd, 'logs', name) + '.log'
        self.conf_filename      = join(analysis.cwd, 'confs', name) + '.py'
        self.service_filename   = join(analysis.cwd, 'fs', name) + '.root'
        self.jobinfo_filename   = join(analysis.cwd, 'report', name) + '.ini'
        self.try_reuse_data     = try_reuse_data
        self.subprocess         = None
        self.time_start         = None
        self.time_end           = None
        self.message = monitor.connect_object_with_messenger(self)

    def __str__(self):
        return "CmsRunProcess(" + self.name + ")"

    def __repr__(self):
        return str(self)

    def prepare_run_conf(self,
                         use_file_service,
                         output_module_name,
                         common_builtins):
        """
        Takes all infos about the cmsRun to be started and builds a
        configuration file with python code, which is passed to cmsRun on
        calling start().
        """

        # collect lines to write out at once.
        conf_lines = [
            "# generated",
            "# on " + time.ctime(),
            "# by cmsrunprocess.py",
            ""
        ]

        # set __builtin__ variables
        smpl = self.sample
        builtin_dict = {
            "lumi"      : smpl.lumi,
            "is_data"   : smpl.is_data,
            "legend"    : smpl.legend,
            "sample"    : smpl.name
        }
        builtin_dict.update(common_builtins)
        builtin_dict.update(smpl.cmsRun_builtin)

        # builtin, imports
        conf_lines += [
            "import __builtin__",
            "__builtin__.cms_var = %s" % repr(builtin_dict),
            "",
            "from " + self.cfg_filename + " import *",
            "",

        ]

        # do input filename statements
        conf_lines.append("process.source.fileNames = [")
        for in_file in smpl.input_files:
            if in_file[:5] == "file:":
                files_in_dir = glob.glob(in_file[5:])
                if not files_in_dir:
                    self.message(
                        "WARNING: no input files globbed for "+in_file[5:]
                    )
                    conf_lines.append("    '" + in_file.strip() + "',")
                else:
                    for fid in files_in_dir:
                        conf_lines.append("    'file:" + fid + "',")
            else:
                conf_lines.append("    '" + in_file.strip() + "',")
        conf_lines.append("]")

        # do output filename statements
        if smpl.output_file:
            filename = smpl.output_file
            if filename[-5:] != ".root":
                filename += self.name + ".root"
            conf_lines.append(
                "process."
                + output_module_name
                + ".fileName = '"
                + filename.strip()
                + "'"
            )

        # fileService statement
        if use_file_service:
            conf_lines.append(
                "process.TFileService.fileName = '"
                + self.service_filename
            )
            conf_lines.append("")

        # custom code
        conf_lines += smpl.cmsRun_add_lines

        # write out file
        with open(self.conf_filename, "w") as conf_file:
            for line in conf_lines:
                conf_file.write(line + "\n")

    def write_job_info(self, exit_code):
        """
        Writes start- and endtime as well as exitcode to the process info file
        in settings.DIR_JOBINFO.
        If self.sigint is true, it does not write anything.
        """
        if settings.recieved_sigint:
            return

        with open(self.jobinfo_filename, "w") as info_file:
            json.dump(
                {
                    "startTime": self.time_start,
                    "endTime": self.time_end,
                    "exitCode": str(exit_code),
                },
                info_file
            )

    def check_reuse_possible(self, check_for_file_service):
        """
        Checks if log, conf and file service files are present and if the
        process was finished successfully before. If yes returns True,
        because the previous results can be used again.
        """
        if not self.try_reuse_data:
            return False
        if not os.path.exists(self.log_filename):
            return False
        if not os.path.exists(self.conf_filename):
            return False
        if (check_for_file_service and
                not os.path.exists(self.service_filename)):
            return False
        if not os.path.exists(self.jobinfo_filename):
            return False
        with open(self.jobinfo_filename) as info_file:
            info = json.load(info_file)
            if type(info) == dict and not int(info.get("exitCode", 255)):
                return True

    def successful(self):
        return (
            self.time_end
            and self.subprocess.returncode == 0
            and not settings.recieved_sigint
        )

    def finalize(self):
        self.time_end = time.ctime()
        if self.subprocess:
            if self.subprocess.returncode == 0 and self.log_file:
                self.log_file.close()
                self.log_file = None
                with open(self.log_filename, "r") as f:
                    if 'Exception ------' in "".join(f.readlines()):
                        self.subprocess.returncode = -1
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.write_job_info(self.subprocess.returncode)

    def start(self):
        """
        Start cmsRun with conf-file.
        """
        self.time_start = time.ctime()

        # delete jobinfo file
        if os.path.exists(self.jobinfo_filename):
            os.remove(self.jobinfo_filename)

        # aaand go for it!
        self.log_file = open(self.log_filename, "w")
        self.subprocess = subprocess.Popen(
            ["cmsRun", self.conf_filename]+self.sample.cmsRun_args,
            stdout=self.log_file,
            stderr=subprocess.STDOUT
        )

    def terminate(self):
        self.subprocess.terminate()


class CmsRunProxy(toolinterface.Tool):
    """Tool to embed cmsRun execution into varial toolchains."""

    def __init__(self,
                 cfg_filename,
                 use_file_service=True,
                 output_module_name="out",
                 common_builtins=None,
                 name=None):
        super(CmsRunProxy, self).__init__(name)
        self.waiting_pros = []
        self.running_pros = []
        self.finished_pros = []
        self.failed_pros = []
        self.cfg_filename = cfg_filename
        self.use_file_service = use_file_service
        self.output_module_name = output_module_name
        self.common_builtins = common_builtins or {}
        self.try_reuse = settings.try_reuse_results

    def wanna_reuse(self, all_reused_before_me):
        self._setup_processes()

        if settings.only_reload_results:
            return True

        return not bool(self.waiting_pros)

    def reuse(self):
        super(CmsRunProxy, self).reuse()
        self._finalize()

    def run(self):
        if settings.suppress_eventloop_exec:
            self.message(
                self, "INFO settings.suppress_eventloop_exec == True, pass...")
            return
        if not (settings.not_ask_execute or raw_input(
                "Really run these cmsRun jobs:\n   "
                + ",\n   ".join(map(str, self.waiting_pros))
                + ('\nusing %i cores' % settings.max_num_processes)
                + "\n?? (type 'yes') "
                ) == "yes"):
            return

        self._handle_processes()
        sig_term_sent = False
        while self.running_pros:
            if settings.recieved_sigint and not sig_term_sent:
                self.abort_all_processes()
                sig_term_sent = True
            time.sleep(0.2)
            self._handle_processes()

        self.result = wrappers.Wrapper(
            finished_procs=list(p.name for p in self.finished_pros))
        self._finalize()

    def _setup_processes(self):
        for d in ('logs', 'confs', 'fs', 'report'):
            path = join(self.cwd, d)
            if not os.path.exists(path):
                os.mkdir(path)

        for name, smpl in analysis.all_samples.iteritems():
            process = CmsRunProcess(smpl, self.try_reuse, self.cfg_filename)
            if process.check_reuse_possible(self.use_file_service):
                self.finished_pros.append(process)
            else:
                self.waiting_pros.append(process)
                monitor.proc_enqueued(process)

    def _handle_processes(self):
        # start processing
        if (len(self.running_pros) < settings.max_num_processes
                and self.waiting_pros):
            process = self.waiting_pros.pop(0)
            process.prepare_run_conf(
                self.use_file_service,
                self.output_module_name,
                self.common_builtins
            )
            process.start()
            monitor.proc_started(process)
            self.running_pros.append(process)

        # finish processes
        for process in self.running_pros[:]:
            process.subprocess.poll()
            if None == process.subprocess.returncode:
                continue

            self.running_pros.remove(process)
            process.finalize()
            if process.successful():
                self.finished_pros.append(process)
                monitor.proc_finished(process)
            else:
                self.failed_pros.append(process)
                monitor.proc_failed(process)

    def _finalize(self):
        if settings.recieved_sigint:
            return
        if not self.use_file_service:
            return
        for process in self.finished_pros:
            analysis.fs_aliases += list(
                alias for alias in diskio.generate_fs_aliases(
                    process.service_filename,
                    process.sample
                )
            )

    def abort_all_processes(self):
        self.waiting_pros = []
        for process in self.running_pros:
            process.terminate()


# TODO think about multiprocessing, in a dedicated light process